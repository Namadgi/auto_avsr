import torch
import torchaudio
from torch import nn

from data_transforms import TextTransform
from espnet_batch_beam_search import BatchBeamSearch

from espnet_length_bonus import LengthBonus
from espnet_ctc import CTCPrefixScorer

from e2e_asr_conformer import E2E

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

class ModelModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list))

        self.model.to(self.device)
        

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


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

