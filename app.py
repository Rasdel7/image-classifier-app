import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import json
import urllib.request
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Classification App")
st.markdown("Classify any image into 1000 categories "
            "using ResNet50 — a pretrained deep "
            "learning model.")
st.markdown("---")

# Load ImageNet class labels
@st.cache_data
def load_labels():
    url = ("https://raw.githubusercontent.com/"
           "anishathalye/imagenet-simple-labels/"
           "master/imagenet-simple-labels.json")
    try:
        with urllib.request.urlopen(url) as f:
            labels = json.load(f)
        return labels
    except:
        # Fallback basic labels
        return [f"class_{i}"
                for i in range(1000)]

# Load ResNet50 model
@st.cache_resource
def load_model():
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# Predict
def predict(image, model, labels, top_k=10):
    with torch.no_grad():
        tensor    = preprocess_image(image)
        outputs   = model(tensor)
        probs     = torch.nn.functional\
                        .softmax(outputs[0],
                                 dim=0)
        top_probs, top_indices = \
            torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(
        top_probs.numpy(),
        top_indices.numpy()
    ):
        results.append({
            'label':       labels[idx]
                           if idx < len(labels)
                           else f"class_{idx}",
            'probability': float(prob),
            'index':       int(idx)
        })
    return results

# Load resources
with st.spinner("Loading ResNet50 model..."):
    model  = load_model()
    labels = load_labels()
st.success(
    "✅ ResNet50 loaded — "
    "1000 ImageNet categories")

# Sidebar
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Top K predictions:",
    3, 20, 10, 1
)
show_all = st.sidebar.checkbox(
    "Show all predictions table",
    value=False
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.info("""
**ResNet50**
- 50 layers deep
- 25.6M parameters
- Trained on ImageNet
- 1000 categories
- Top-5 accuracy: 93.7%
""")

st.sidebar.markdown("### 📂 Categories Include")
sample_cats = [
    "🐕 Dogs (120 breeds)",
    "🐈 Cats (various)",
    "🚗 Vehicles",
    "🍎 Food items",
    "🏠 Buildings",
    "🌿 Plants & flowers",
    "🐦 Birds (many species)",
    "👔 Clothing",
    "🔧 Tools & instruments",
    "1000 total classes..."
]
for cat in sample_cats:
    st.sidebar.markdown(f"• {cat}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Classify Image",
    "📊 Prediction Details",
    "🔬 Model Visualization",
    "📚 About ResNet"
])

# Session state
if 'predictions'  not in st.session_state:
    st.session_state.predictions  = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Tab 1 — Classify
with tab1:
    st.markdown("### Upload Image to Classify")

    uploaded = st.file_uploader(
        "Upload any image:",
        type=['jpg', 'jpeg', 'png',
              'webp', 'bmp']
    )

    if uploaded:
        image = Image.open(uploaded)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📷 Your Image")
            st.image(image,
                     use_column_width=True)
            st.caption(
                f"Size: {image.size[0]}×"
                f"{image.size[1]} pixels")

        with col2:
            st.markdown("#### 🎯 Top Prediction")
            with st.spinner(
                "Classifying with ResNet50..."
            ):
                predictions = predict(
                    image, model,
                    labels, top_k)

            st.session_state.predictions = \
                predictions
            st.session_state.current_image = \
                image

            top = predictions[0]
            st.markdown(
                f"<h2 style='color:#3498db;"
                f"text-align:center'>"
                f"🏷️ {top['label'].title()}"
                f"</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h3 style='color:#2ecc71;"
                f"text-align:center'>"
                f"{top['probability']:.1%} "
                f"confidence</h3>",
                unsafe_allow_html=True
            )

            # Top 5 chart
            top5    = predictions[:5]
            labels5 = [p['label'][:20]
                       for p in top5]
            probs5  = [p['probability']
                       for p in top5]

            fig, ax = plt.subplots(
                figsize=(7, 4))
            colors  = [
                '#3498db', '#2ecc71',
                '#f39c12', '#e74c3c',
                '#9b59b6'
            ]
            bars = ax.barh(
                labels5[::-1],
                probs5[::-1],
                color=colors[::-1],
                edgecolor='black'
            )
            for bar, val in zip(
                bars, probs5[::-1]
            ):
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() +
                    bar.get_height()/2,
                    f'{val:.1%}',
                    va='center',
                    fontsize=10,
                    fontweight='bold'
                )
            ax.set_xlim(0, 1.15)
            ax.set_title(
                'Top 5 Predictions',
                fontsize=13)
            ax.set_xlabel(
                'Probability')
            plt.tight_layout()
            st.pyplot(fig)

# Tab 2 — Prediction Details
with tab2:
    st.markdown(
        "### 📊 Full Prediction Details")

    if st.session_state.predictions:
        import pandas as pd

        preds = st.session_state.predictions
        df    = pd.DataFrame([{
            'Rank':        i + 1,
            'Label':       p['label'].title(),
            'Probability': f"{p['probability']:.4%}",
            'Class Index': p['index']
        } for i, p in enumerate(preds)])

        st.dataframe(df,
                     use_container_width=True,
                     hide_index=True)

        # Full probability chart
        fig2, ax2 = plt.subplots(
            figsize=(10, 6))
        all_labels = [
            p['label'][:25]
            for p in preds
        ]
        all_probs  = [
            p['probability']
            for p in preds
        ]
        bar_colors = [
            '#3498db' if i == 0
            else '#95a5a6'
            for i in range(len(preds))
        ]
        ax2.barh(
            all_labels[::-1],
            all_probs[::-1],
            color=bar_colors[::-1],
            edgecolor='black'
        )
        ax2.set_title(
            f'Top {len(preds)} '
            f'Predictions — Probability',
            fontsize=13)
        ax2.set_xlabel('Probability')
        plt.tight_layout()
        st.pyplot(fig2)

        # Confidence interpretation
        top_prob = preds[0]['probability']
        st.markdown("### 🎯 Confidence Analysis")

        if top_prob >= 0.9:
            st.success(
                f"🟢 Very High Confidence "
                f"({top_prob:.1%}) — "
                f"Model is very certain")
        elif top_prob >= 0.7:
            st.success(
                f"🟢 High Confidence "
                f"({top_prob:.1%}) — "
                f"Good prediction")
        elif top_prob >= 0.5:
            st.warning(
                f"🟡 Moderate Confidence "
                f"({top_prob:.1%}) — "
                f"Reasonable prediction")
        elif top_prob >= 0.3:
            st.warning(
                f"🟡 Low Confidence "
                f"({top_prob:.1%}) — "
                f"Uncertain prediction")
        else:
            st.error(
                f"🔴 Very Low Confidence "
                f"({top_prob:.1%}) — "
                f"Image may not be in "
                f"ImageNet categories")
    else:
        st.info(
            "Upload an image in the "
            "Classify tab first!")

# Tab 3 — Model Visualization
with tab3:
    st.markdown("### 🔬 ResNet50 Architecture")

    st.markdown("""
    #### Network Structure
    ResNet50 uses **residual connections**
    (skip connections) that allow gradients
    to flow directly through the network,
    solving the vanishing gradient problem.
    """)

    import pandas as pd
    arch_df = pd.DataFrame({
        'Layer':       [
            'Input',
            'Conv1 + MaxPool',
            'Layer 1 (3 blocks)',
            'Layer 2 (4 blocks)',
            'Layer 3 (6 blocks)',
            'Layer 4 (3 blocks)',
            'AvgPool + FC'
        ],
        'Output Shape': [
            '3 × 224 × 224',
            '64 × 56 × 56',
            '256 × 56 × 56',
            '512 × 28 × 28',
            '1024 × 14 × 14',
            '2048 × 7 × 7',
            '1000'
        ],
        'Parameters': [
            '—',
            '9,472',
            '215,808',
            '1,219,584',
            '7,098,368',
            '14,964,736',
            '2,049,000'
        ]
    })
    st.dataframe(arch_df,
                 use_container_width=True,
                 hide_index=True)

    # Preprocessing visualization
    if st.session_state.current_image:
        st.markdown(
            "#### 🔄 Image Preprocessing")
        st.markdown(
            "ResNet requires images to be "
            "resized, cropped and normalized "
            "before classification.")

        img   = st.session_state.current_image
        cols  = st.columns(3)

        with cols[0]:
            st.image(img, caption="Original",
                     use_column_width=True)

        with cols[1]:
            resized = img.resize((256, 256))
            st.image(resized,
                     caption="Resized (256×256)",
                     use_column_width=True)

        with cols[2]:
            w, h     = resized.size
            left     = (w - 224) // 2
            top      = (h - 224) // 2
            cropped  = resized.crop(
                (left, top,
                 left + 224, top + 224))
            st.image(cropped,
                     caption="Cropped (224×224)",
                     use_column_width=True)

# Tab 4 — About
with tab4:
    st.markdown("### 📚 About ResNet50")
    st.markdown("""
    #### What is ResNet?
    **ResNet (Residual Network)** was introduced
    by Microsoft Research in 2015 and won the
    ImageNet competition with 3.57% top-5 error.

    #### The Key Innovation — Skip Connections
    Traditional deep networks suffer from
    **vanishing gradients** — the signal
    weakens as it travels through many layers.

    ResNet solves this with **residual blocks**:
                Output = F(x) + x
                The input x is added directly to the output,
    creating a shortcut that lets gradients flow
    freely through the network.

    #### ImageNet Dataset
    - 1.2 million training images
    - 1000 categories
    - Annual competition since 2010
    - Benchmark for image classification

    #### Transfer Learning
    Instead of training from scratch
    (weeks on GPUs), we use weights
    pre-trained on ImageNet and apply
    them to new images instantly.
    This is called **transfer learning**
    — one of the most powerful concepts
    in modern AI.
    """)

    import pandas as pd
    comparison = pd.DataFrame({
        'Model':      ['AlexNet', 'VGG16',
                       'ResNet50', 'ResNet152',
                       'EfficientNet-B7'],
        'Year':       [2012, 2014,
                       2015, 2015, 2019],
        'Parameters': ['60M', '138M',
                       '25.6M', '60M', '66M'],
        'Top-5 Acc':  ['84.6%', '92.7%',
                       '93.7%', '94.6%',
                       '97.1%'],
        'Depth':      [8, 16, 50, 152, 813]
    })
    st.markdown("#### 📊 Model Comparison")
    st.dataframe(comparison,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Image Classification using ResNet50 | "
    "Trained on ImageNet — 1000 categories"
)