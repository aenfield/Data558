from shared_routines import *
from PIL import Image, ImageOps

# This is the very simplest first cut at image augmentation - only does mirroring (could be expanded), doesn't
# intelligently handle cases where images might already be augmented (so it doesn't generate new mirrors of
# already mirrored images, for example), etc. But it's a start, and grabs the lowest-hanging fruit. Another
# improvement could be to output the files to a separate mirrored directory structure which would be easier
# to delete and re-run, this would perhaps require modifying the images_to_features code in order to look
# at multiple directories. For now, this is good.

def main():
    list_images = get_image_list('train')

    for image_path in list_images:
        imgbytes = open(image_path, 'rb')
        img = Image.open(imgbytes)

        img_mirrored = ImageOps.mirror(img)

        new_image_path = "{}-m.jpg".format(image_path.rstrip('.jpg'))
        img_mirrored.save(new_image_path)
        print("Saved {}.".format(new_image_path))


if __name__ == "__main__":
   main()

