import argparse
import os
import pickle
import matplotlib.pyplot as plt
import torch
import preprocessing as pre
import model
from vocab import Vocab, label_to_index, indices_to_latex
from model import CRNN, init_weights

def main_routine(img_paths, model_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    vocab = Vocab()
    vocab = pickle.load(open('../saved/vocab', 'rb'))

    model = CRNN(vocab, embed_size=32)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    image_sizes = [
        (40, 160), (40, 200), (40, 240), (40, 280), (40, 320), \
        (50, 120), (50, 200), (50, 240), (50, 280), (50, 320)
    ]
    with torch.no_grad():
        for path in img_paths:
            img, _, _ = pre.reshape_images(path, image_sizes=image_sizes)
            tensor_img = torch.unsqueeze(torch.tensor(img), 0).to(device)
            output = model(tensor_img, teacher_forcing=False)
            #plt.imshow(plt.imread(path), cmap='gray')
            #plt.show()()
            print(indices_to_latex(output, vocab))
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', 
        help="If given, searches for images in <path>. Else, searches in ../images",
        type=str
    )
    parser.add_argument(
        '--model',
        help="Path for model parameters; ../saved/params/CRNN_params_epoch_3.pt if not given.",
        type=str
    )
    args = parser.parse_args()

    img_dir = args.path if args.path else '../images'
    img_paths = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if os.path.isfile(os.path.join(img_dir, f))
    ]
    model_path = args.model if args.model else '../saved/params/CRNN_params_epoch_3.pt'
    
    main_routine(img_paths, model_path)
