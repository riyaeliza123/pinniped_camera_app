# Handles display of annotated images

import supervision as sv

def create_annotated_image(image_np, detections, result_json):
    """Return annotated image (NumPy array) for display only."""
    if len(detections.xyxy) == 0:
        return image_np

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    labels = [item["class"] for item in result_json["predictions"]]
    annotated_image = box_annotator.annotate(scene=image_np, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image
