import torch
import torchaudio
from torch import nn
import time
import logging
from contextlib import nullcontext

from src.ml_models.data_transforms import TextTransform
from src.ml_models.espnet_batch_beam_search import BatchBeamSearch
from src.ml_models.espnet_length_bonus import LengthBonus
from src.ml_models.espnet_ctc import CTCPrefixScorer
from src.ml_models.e2e_asr_conformer import E2E
from src.ml_models.timing import StageTimer

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

class ModelModule(nn.Module):
    """Conformer encoder + decoder wrapper.

    Defaults reproduce the production configuration (beam_size=40, ctc_weight=0.1,
    attention/CTC joint beam search, fp32). `decode_mode="ctc_greedy"` bypasses the
    autoregressive decoder entirely and uses the CTC head's argmax path.
    """

    def __init__(self, device='cuda', beam_size=40, ctc_weight=0.1, decode_mode='beam',
                 autocast_dtype=None, timer: StageTimer = None):
        super().__init__()
        self.device = device
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.decode_mode = decode_mode
        if isinstance(autocast_dtype, str):
            autocast_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}[autocast_dtype]
        self.autocast_dtype = autocast_dtype
        self.timer = timer or StageTimer(device, sync=False)

        self.text_transform = TextTransform(
            sp_model_path="extra_files/unigram5000.model",
            dict_path="extra_files/unigram5000_units.txt"
        )
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list))

        self.model.to(self.device)
        # BatchBeamSearch is stateless between calls (per-request state travels in the
        # hypotheses), so it is safe to build once instead of on every forward().
        # It is an nn.Module that re-registers model.decoder/model.ctc under nn_dict.*, so
        # keep it out of this module's state_dict (checkpoints must keep prod's key layout).
        self.__dict__['beam_search'] = get_beam_search_decoder(
            self.model, self.token_list, ctc_weight=self.ctc_weight, beam_size=self.beam_size
        )

    def _autocast(self):
        if self.autocast_dtype is None or not str(self.device).startswith('cuda'):
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.autocast_dtype)

    def forward(self, sample):
        with self.timer.stage('encoder'), self._autocast():
            cur_time = time.time()
            enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
            enc_feat = enc_feat.squeeze(0)
            logging.info(f"Encoder: {time.time() - cur_time:.4f}s")

        with self.timer.stage('decode'):
            cur_time = time.time()
            if self.decode_mode == 'ctc_greedy':
                predicted_token_id = self._ctc_greedy(enc_feat)
            else:
                with self._autocast():
                    nbest_hyps = self.beam_search(enc_feat)
                nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
                predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            logging.info(f"Beam search: {time.time() - cur_time:.4f}s")

        cur_time = time.time()
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        logging.info(f"Post process: {time.time() - cur_time:.4f}s")

        return predicted

    def _ctc_greedy(self, enc_feat):
        """CTC greedy decoding: argmax per frame, collapse repeats, drop blanks (id 0)."""
        ids = self.model.ctc.argmax(enc_feat.unsqueeze(0)).squeeze(0)
        ids = torch.unique_consecutive(ids)
        ids = ids[ids != 0]
        return ids.cpu()


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
