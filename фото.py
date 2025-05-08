from PIL import Image
import matplotlib.pyplot as plt

# Загрузка изображения и принудительное приведение к RGB
im_RGB = Image.open(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\lion.png").convert('RGB')

# Преобразования
im_GS  = im_RGB.convert("L")                                # Grayscale
im_D   = im_RGB.convert("1")                                # B/W с дизерингом
im_WD  = im_RGB.convert("1", dither=Image.Dither.NONE)   # B/W без дизеринга

# Сохранение

# im_GS.save(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\cat_gray.png")
# im_D.save(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\cat_bw.png")
# im_RGB.save(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\Lenna.png")
# im_GS.save(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\Lenna_gray.png")
# im_D.save(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\Lenna_bw.png")
# im_WD.save(r"C:\ЛЭТИ 2 курс\АиСД 4 сем\лаб2\Lenna_bw_no_dither.png")
im_RGB.save("lion.png")
im_GS.save("lion_gray.png")
im_D.save("lion_bw.png")
im_WD.save("lion_bw_no_dither.png")

# Подготовка для отображения
images = [im_RGB, im_GS, im_D, im_WD]
titles = ["Оригинал (RGB)", "Grayscale", "B/W с дизерингом", "B/W без дизеринга"]

# Отображение
plt.figure(figsize=(14, 4))
for i in range(len(images)):
    img = images[i]
    title = titles[i]
    plt.subplot(1, 4, i + 1)
    if img.mode in ("L", "1"):
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title(title)

    plt.axis("off")

plt.tight_layout()
plt.show()
