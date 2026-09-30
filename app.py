from __future__ import annotations

from io import BytesIO
from pathlib import Path
from PIL import Image
import gradio as gr

from src.config import ASSETS_DIR
from src.inference import predict_image_bytes


def analyze_xray(input_image):
    if input_image is None:
        return None, "⚠️ Please upload a chest X-ray image to get started."

    # Convert PIL Image to bytes for inference pipeline
    buffer = BytesIO()
    if isinstance(input_image, Image.Image):
        img = input_image.convert("RGB")
    else:
        img = Image.fromarray(input_image).convert("RGB")
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # Run inference
    result = predict_image_bytes(image_bytes)

    # Prepare probability dict for Gradio Label
    probabilities = {
        label: float(prob)
        for label, prob in result.probabilities.items()
    }

    # Format status text
    status_icon = "🩺" if result.label == "PNEUMONIA" else "✅"
    confidence_pct = result.confidence * 100.0

    markdown_output = f"""
### {status_icon} Predicted Diagnosis: **{result.label}**
- **Confidence Score:** `{confidence_pct:.2f}%`
- **Inference Mode:** `{result.mode.replace('_', ' ').title()}`
- **Model Architecture:** `MobileNetV2 (Transfer Learning)`

> **ℹ️ Note:** {result.note}
"""

    return probabilities, markdown_output


# Load sample examples if available
example_images = []
normal_sample = ASSETS_DIR / "sample_normal.png"
pneumonia_sample = ASSETS_DIR / "sample_pneumonia.png"

if normal_sample.exists():
    example_images.append([str(normal_sample)])
if pneumonia_sample.exists():
    example_images.append([str(pneumonia_sample)])


custom_css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: 'Inter', sans-serif !important;
}
#header-box {
    text-align: center;
    margin-bottom: 20px;
}
"""

with gr.Blocks(title="Pneumonia Detection System | REVA University", css=custom_css, theme=gr.themes.Soft(primary_hue="orange")) as demo:
    with gr.Column(elem_id="header-box"):
        gr.Markdown(
            """
            # 🩺 Pneumonia Detection System
            ### AI-Powered Chest X-Ray Classification using MobileNetV2
            **REVA University** | B.Tech CSE (AIML) Project by **Madan S**
            """
        )
        gr.Markdown("Upload a chest X-ray scan below or click on any of the example scans to test the model.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📤 Upload Chest X-Ray Scan (PNG / JPG)",
                sources=["upload", "clipboard"]
            )
            submit_btn = gr.Button("🔍 Analyze Chest X-Ray", variant="primary", size="lg")
            
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="📁 Sample Test Scans"
                )

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="📊 Class Probabilities",
                num_top_classes=2
            )
            status_output = gr.Markdown(
                """
                ### ⏳ Status: Ready
                Upload an image and click **Analyze** to see real-time classification results.
                """
            )

    submit_btn.click(
        fn=analyze_xray,
        inputs=image_input,
        outputs=[label_output, status_output]
    )
    
    image_input.change(
        fn=analyze_xray,
        inputs=image_input,
        outputs=[label_output, status_output]
    )

    gr.Markdown(
        """
        ---
        > ⚠️ **Disclaimer:** This tool is designed strictly for academic and portfolio demonstration purposes. It is **not** a certified clinical diagnostic device. Always consult certified medical practitioners for clinical assessments.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
