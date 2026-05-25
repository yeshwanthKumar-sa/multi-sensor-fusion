from PIL import ImageDraw

def draw_boxes(image):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    width, height = image.size

    vehicle_box = [
        (width * 0.15, height * 0.55),
        (width * 0.45, height * 0.9)
    ]

    pedestrian_box = [
    (width * 0.78, height * 0.42),
    (width * 0.88, height * 0.78)
    ]

    draw.rectangle(
        vehicle_box,
        outline="red",
        width=5
    )

    draw.text(
        (vehicle_box[0][0], vehicle_box[0][1] - 30),
        "Vehicle",
        fill="red"
    )

    draw.rectangle(
        pedestrian_box,
        outline="blue",
        width=5
    )

    draw.text(
        (pedestrian_box[0][0], pedestrian_box[0][1] - 30),
        "Pedestrian",
        fill="blue"
    )

    return image