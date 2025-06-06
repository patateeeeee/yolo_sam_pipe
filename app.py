import gradio as gr
from PIL import Image
from yolo_utils import apply_model, get_labels
from sam_utils import apply_SAM_multi

# Detection + segmentation in a single button
def detect_and_segment(input_image: Image.Image, selected_label: str):
    threshold = 0.3  # Fixed threshold
    img_out, boxes, labels = apply_model(input_image, threshold, return_labels=True)
    filtered_boxes = [b for b, l in zip(boxes, labels) if l == selected_label or selected_label == "All"]
    if not filtered_boxes:
        masks = [input_image]
    else:
        masks = apply_SAM_multi(input_image, filtered_boxes)
    return masks

labels_list = list(get_labels().values())
labels_list = ["All"] + labels_list

with gr.Blocks() as demo:
    gr.Markdown("# Yolo Detection + Segmentation with SAM2")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📤 Input image")
            label_dropdown = gr.Dropdown(choices=labels_list, value="All", label="🎯 Label to detect")
            process_btn = gr.Button("Detect and segment")

        with gr.Column():
            segmented_gallery = gr.Gallery(label="🧼 SAM Masks", columns=2)

    process_btn.click(
        fn=detect_and_segment,
        inputs=[input_image, label_dropdown],
        outputs=segmented_gallery
    )

if __name__ == "__main__":
    demo.launch()
