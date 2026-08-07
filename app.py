import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd


# ===================== PAGE SETTINGS =====================

st.set_page_config(
    page_title="AI Flower Recognition",
    page_icon="🌸",
    layout="centered"
)


# ===================== LOAD MODEL =====================

@st.cache_resource
def load_ai_model():
    return load_model("flower_recognition_model.keras")


model = load_ai_model()


# ===================== FLOWER CLASSES =====================

class_names = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]


flower_info = {

    "Daisy":
    "🌼 Daisy is a cheerful flower that symbolizes innocence and purity. It is commonly found in gardens and grasslands.",

    "Dandelion":
    "🌼 Dandelion is known for its bright yellow flowers and fluffy seed heads. It symbolizes hope, resilience, and new beginnings.",

    "Rose":
    "🌹 Rose is one of the world's most popular flowers and represents love, beauty, and passion. It comes in many beautiful colors.",

    "Sunflower":
    "🌻 Sunflower is famous for turning towards the sun. It symbolizes happiness, positivity, and loyalty.",

    "Tulip":
    "🌷 Tulip is a colorful spring flower that symbolizes elegance and perfect love."
}


flower_details = {

    "Daisy": {
        "Scientific Name": "Bellis perennis",
        "Flower Color": "White with Yellow Center",
        "Native Region": "Europe",
        "Bloom Season": "Spring to Summer"
    },

    "Dandelion": {
        "Scientific Name": "Taraxacum officinale",
        "Flower Color": "Yellow",
        "Native Region": "Europe and Asia",
        "Bloom Season": "Spring"
    },

    "Rose": {
        "Scientific Name": "Rosa",
        "Flower Color": "Red, Pink, White, Yellow",
        "Native Region": "Asia",
        "Bloom Season": "Spring to Fall"
    },

    "Sunflower": {
        "Scientific Name": "Helianthus annuus",
        "Flower Color": "Yellow",
        "Native Region": "North America",
        "Bloom Season": "Summer"
    },

    "Tulip": {
        "Scientific Name": "Tulipa",
        "Flower Color": "Red, Pink, Yellow, White, Purple",
        "Native Region": "Central Asia",
        "Bloom Season": "Spring"
    }

}


flower_emoji = {
    "Daisy": "🌼",
    "Dandelion": "🌼",
    "Rose": "🌹",
    "Sunflower": "🌻",
    "Tulip": "🌷"
}


# ===================== SIDEBAR =====================

st.sidebar.title("🌸 AI Flower Recognition")


st.sidebar.markdown("---")


st.sidebar.markdown("## 📌 About Project")

st.sidebar.info(
"""
This application uses a Convolutional Neural Network (CNN)
to identify flower species from uploaded images.

Upload a flower image and AI will predict the flower
with confidence score.
"""
)


st.sidebar.markdown("---")


st.sidebar.markdown("## ⚡ Quick Facts")


col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric("🌸 Classes", "5")

with col2:
    st.metric("🤖 Model", "CNN")


st.sidebar.metric("📐 Input Size", "224 × 224")


st.sidebar.markdown("---")


st.sidebar.markdown("## 🌼 Recognizes")

st.sidebar.write(
"""
🌼 Daisy

🌼 Dandelion

🌹 Rose

🌻 Sunflower

🌷 Tulip
"""
)


st.sidebar.markdown("---")


st.sidebar.markdown("## 🛠️ Technologies")

st.sidebar.write(
"""
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
"""
)
# ===================== MAIN UI =====================

st.markdown(
"""
<h1 style="text-align:center;color:#E91E63;">
🌸 AI Flower Recognition System
</h1>

<h3 style="text-align:center;">
Identify Flower Species using Deep Learning (CNN)
</h3>
""",
unsafe_allow_html=True
)


st.markdown("---")


uploaded_file = st.file_uploader(
    "📤 Upload a Flower Image",
    type=["jpg", "jpeg", "png"]
)


st.info(
"""
📌 Upload a clear image of a single flower.

The model can recognize:
Daisy, Dandelion, Rose, Sunflower and Tulip.
"""
)



if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Flower Image",
        use_container_width=True
    )


    if st.button("🔍 Predict Flower"):


        img = image.load_img(
            uploaded_file,
            target_size=(224,224)
        )


        img_array = image.img_to_array(img)


        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # Same preprocessing as Colab
        img_array = img_array / 255.0


        prediction = model.predict(img_array)


        predicted_index = np.argmax(prediction)


        confidence = np.max(prediction) * 100


        flower_name = class_names[predicted_index]



        if confidence < 70:

            st.warning(
                "⚠️ Low confidence prediction. Please upload a clearer flower image."
            )

            st.metric(
                "🎯 Confidence",
                f"{confidence:.2f}%"
            )


        else:

            st.success("✅ Prediction Completed!")


            # ================= RESULT CARD =================

            st.markdown(
            f"""
            <div style="
            background:linear-gradient(135deg,#E91E63,#F06292);
            padding:25px;
            border-radius:18px;
            color:white;
            ">

            <h1 style="text-align:center;">
            🎉 Prediction Result
            </h1>

            <hr>

            <h3>
            🌸 Flower:
            {flower_name} {flower_emoji[flower_name]}
            </h3>

            <h3>
            🎯 Confidence:
            {confidence:.2f}%
            </h3>

            <h3>
            🤖 Model:
            CNN (TensorFlow)
            </h3>

            </div>
            """,
            unsafe_allow_html=True
            )



            # ================= FLOWER INFO =================

            st.markdown(
f"""
<div style="
background:#FFF8E1;
padding:20px;
border-radius:15px;
border-left:6px solid #FFC107;
">

<h2 style="color:#D81B60;">
📖 About {flower_name} {flower_emoji[flower_name]}
</h2>

<p style="
font-size:17px;
color:#333333;
line-height:1.8;
">
{flower_info[flower_name]}
</p>

</div>
""",
unsafe_allow_html=True
)

            # ================= DETAILS =================

            st.markdown("## 🌿 Flower Details")


            details = flower_details[flower_name]


            col1, col2 = st.columns(2)


            with col1:

                st.success(
                    f"🔬 Scientific Name\n\n{details['Scientific Name']}"
                )

                st.info(
                    f"🎨 Color\n\n{details['Flower Color']}"
                )


            with col2:

                st.warning(
                    f"🌍 Native Region\n\n{details['Native Region']}"
                )

                st.success(
                    f"🌼 Bloom Season\n\n{details['Bloom Season']}"
                )



            # ================= PROBABILITY =================

            st.markdown("## 📊 Prediction Probability")


            probabilities = prediction[0] * 100


            df_prob = pd.DataFrame(
            {
                "Flower": class_names,
                "Confidence": probabilities
            }
            )


            df_prob = df_prob.sort_values(
                by="Confidence",
                ascending=False
            )


            st.dataframe(
                df_prob,
                use_container_width=True,
                hide_index=True
            )


            st.bar_chart(
                df_prob.set_index("Flower")
            )