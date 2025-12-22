'''
import random
import os

def recommend_images(skin_tone, base_dir='static/outfit_dataset'):
    outfits = {'male': [], 'female': []}

    for gender in outfits:
        folder = os.path.join(base_dir, gender, skin_tone)
        if os.path.exists(folder):
            imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            outfits[gender] = random.sample(imgs, min(25, len(imgs)))

    return outfits

'''
import random
import os

def recommend_images(skin_tone, base_dir='static/outfit_dataset'):
    outfits = {
        'male': {
            'Bussiness_attire': [],
            'Formal_wear': [],
            'Kurti': [],
            'Wedding': []
        },
        'female': {
            'Bussiness_attire': [],
            'Formal_wear': [],
            'Kurti': [],
            'Wedding': []
        }
    }

    for gender in outfits:
        for category in outfits[gender]:
            folder = os.path.join(base_dir, gender, skin_tone, category)
            if os.path.exists(folder):
                imgs = [
                    f"outfit_dataset/{gender}/{skin_tone}/{category}/{f}"
                    for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                outfits[gender][category] = random.sample(imgs, min(50, len(imgs)))

    return outfits
