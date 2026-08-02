import io

# import bentoml
import torch

from bentoml.io import Image, Text
from PIL import Image as PILImage
# from Xray.ml.model.arch import Net
from Xray.constants.training_pipeline import *
from Xray.ml.model.arch import Net

torch.serialization.add_safe_globals({
    "Xray.ml.model.arch.Net": Net
})

import bentoml



bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)

runner = bento_model.to_runner()

svc = bentoml.Service(
    name=BENTOML_SERVICE_NAME,
    runners=[runner]
)


@svc.api(input=Image(allowed_mime_types=["image/jpeg"]), output=Text())
async def predict(img):

    b = io.BytesIO()

    img.save(b, "jpeg")

    image_bytes = b.getvalue()

    transform = bento_model.custom_objects[TRAIN_TRANSFORMS_KEY]

    image = PILImage.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    result = await runner.async_run(image)

    prediction = torch.argmax(
        result,
        dim=1
    ).cpu().item()

    return PREDICTION_LABEL[prediction]