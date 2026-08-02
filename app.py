import streamlit as st
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from Xray.ml.model.arch import Net
import os


MODEL_PATH = "C:/cnnproject/artifacts/07_31_2026_04_58_05/model_training/model.pt"


# ---------------- Load Model ----------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Net()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


# ---------------- Transform ----------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ---------------- Prediction Function ----------------

def predict_image(image):

    image = image.convert("RGB")

    img = transform(image)

    img = img.unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(img)

        prob = torch.softmax(output, dim=1)

        pred = torch.argmax(prob, dim=1).item()

        confidence = torch.max(prob).item()*100


    classes = {
        0:"NORMAL",
        1:"PNEUMONIA"
    }


    return classes[pred], round(confidence,2)



# ---------------- Streamlit UI ----------------


st.title("🩻 Chest X-Ray Batch Prediction")

st.write(
    "Upload 100-200 X-Ray images and get predictions"
)


uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)



if uploaded_files:


    st.info(
        f"{len(uploaded_files)} images uploaded"
    )


    if st.button("🔍 Predict All"):


        results=[]


        progress = st.progress(0)


        for index,file in enumerate(uploaded_files):

            image = Image.open(file)


            prediction, confidence = predict_image(image)


            results.append(
                {
                    "Image":file.name,
                    "Prediction":prediction,
                    "Confidence":confidence
                }
            )


            progress.progress(
                (index+1)/len(uploaded_files)
            )


        df = pd.DataFrame(results)


        st.success("Prediction Completed")


        st.dataframe(
            df,
            use_container_width=True
        )


        csv = df.to_csv(index=False)


        st.download_button(
            "Download CSV",
            csv,
            "xray_predictions.csv",
            "text/csv"
        )