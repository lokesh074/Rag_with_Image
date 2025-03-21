import fitz
import os

def extract_content_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = []
    image_paths = []

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text("text")
        text_content.append(text)

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image_path = f"images/image_page{page_num+1}_{img_index}.{img_ext}"

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)

    return text_content, image_paths

# pdf_path = "data_analyst.pdf"
# text_data, images = extract_content_from_pdf(pdf_path)
# print("Extracted Text:", text_data[:200])  # Show some extracted text
# print("Extracted Images:", images)
