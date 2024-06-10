import io
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

def save_images_to_pdf(images):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf_name = '_'.join(image_file.name for image_file in images) + '.pdf'

    for image in images:
        img = Image.open(image)
        img_width, img_height = img.size
        scale = min(letter[0] / img_width, letter[1] / img_height)
        img_width *= scale
        img_height *= scale
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        image_reader = ImageReader(img_buffer)
        c.drawImage(image_reader, 0, letter[1] - img_height, width=img_width, height=img_height)
        c.showPage()

    c.save()
    pdf_buffer.seek(0)
    return pdf_name, pdf_buffer

