from google.cloud import aiplatform

# aiplatform.init(
#     # your Google Cloud Project ID or number
#     # environment default used is not set
#     project='biometry-416410',

#     # the Vertex AI region you will use
#     # defaults to us-central1
#     location='us-central1',

#     # Google Cloud Storage bucket in same region as location
#     # used to stage artifacts
#     staging_bucket='gs://biometry',

#     # custom google.auth.credentials.Credentials
#     # environment default credentials used if not set
#     # credentials=my_credentials,

#     # customer managed encryption key resource name
#     # will be applied to all Vertex AI resources if set
#     # encryption_spec_key_name=my_encryption_key_name,

#     # the name of the experiment to use to track
#     # logged metrics and parameters
#     # experiment='my-experiment',

#     # description of the experiment above
#     # experiment_description='my experiment description'
# )

VERSION = 1
APP_NAME = 'vsr'
CUSTOM_PREDICTOR_IMAGE_URI = 'us-central1-docker.pkg.dev/biometry-416410/biometry-docker-repo/vsr:latest'

model_display_name = f"{APP_NAME}-v{VERSION}"
model_description = "PyTorch based video classification with custom container"
MODEL_NAME = 'vsr'
health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [8080]

model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
)
# model = aiplatform.Model.list()[0]
model.wait()

# endpoint = aiplatform.Endpoint.list()[0]
# print(model.display_name)
# print(model.resource_name)

endpoint_display_name = f"{APP_NAME}-endpoint"
endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

# aiplatform.Model.

# traffic_percentage = 100
# machine_type = "n1-standard-4"
# accelerator_type = 'NVIDIA_TESLA_T4'
# accelerator_count = 1
# deployed_model_display_name = model_display_name
# min_replica_count = 1
# max_replica_count = 1
# sync = True

# model.deploy(
#     endpoint=endpoint,
#     deployed_model_display_name=deployed_model_display_name,
#     traffic_percentage=traffic_percentage,
#     machine_type=machine_type,
#     sync=sync,
#     accelerator_type=accelerator_type,
#     accelerator_count=accelerator_count

# )

# model.wait()