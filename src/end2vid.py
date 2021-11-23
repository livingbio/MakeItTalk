import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import util.utils as util
from scipy.signal import savgol_filter
import scipy.io.wavfile

from src.approaches.train_audio2landmark import Audio2landmark_model

from app.conf import settings

class End2vid:
    def __init__(self) -> None:
        self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)

    def run(self, audio, img, audio_name, image_name) -> str:

        ADD_NAIVE_EYE = True

        scipy.io.wavfile.write(f'examples/{audio_name}', audio[0], audio[1])
        predictor = self.predictor
        ''' STEP 1: preprocess input single image '''
        shapes = predictor.get_landmarks(img)
        if (not shapes or len(shapes) != 1):
            return 'Cannot detect face landmarks. Exit.'
            # exit(-1)
        shape_3d = shapes[0]

        if(settings.close_input_face_mouth):
            util.close_input_face_mouth(shape_3d)


        ''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
        # shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 0.95 + np.mean(shape_3d[48:, 0])
        shape_3d[49:54, 1] += 1.
        shape_3d[55:60, 1] -= 1.
        shape_3d[[37,38,43,44], 1] -=2
        shape_3d[[40,41,46,47], 1] +=2


        ''' STEP 2: normalize face as input to audio branch '''
        shape_3d, scale, shift = util.norm_input_face(shape_3d)


        ''' STEP 3: Generate audio data as input to audio branch '''
        # audio real data
        au_data = []
        au_emb = []
        # ains = glob.glob1('examples', '*.wav')
        # ains = [item for item in ains if item != 'tmp.wav']
        # ains.sort()
        ains = glob.glob1('examples', audio_name)
        for ain in ains:
            os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
            shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

            # au embedding

            from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
            
            me, ae = get_spk_emb('examples/{}'.format(ain))
            au_emb.append(me.reshape(-1))

            print('Processing audio file', ain)
            c = AutoVC_mel_Convertor('examples')

            au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
                autovc_model_path=settings.load_AUTOVC_name)
            au_data += au_data_i
        if(os.path.isfile('examples/tmp.wav')):
            os.remove('examples/tmp.wav')

        # landmark fake placeholder
        fl_data = []
        rot_tran, rot_quat, anchor_t_shape = [], [], []
        for au, info in au_data:
            au_length = au.shape[0]
            fl = np.zeros(shape=(au_length, 68 * 3))
            fl_data.append((fl, info))
            rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
            rot_quat.append(np.zeros(shape=(au_length, 4)))
            anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
        if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
        if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
            os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

        with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
            pickle.dump(fl_data, fp)
        with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
            pickle.dump(au_data, fp)
        with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
            gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
            pickle.dump(gaze, fp)


        ''' STEP 4: RUN audio->landmark network'''
        model = Audio2landmark_model(settings, jpg_shape=shape_3d)
        if(len(settings.reuse_train_emb_list) == 0):
            model.test(au_emb=au_emb)
        else:
            model.test(au_emb=None)


        ''' STEP 5: de-normalize the output to the original image scale '''
        # fls = glob.glob1('examples', 'pred_fls_*.txt')
        # fls.sort()
        fls = glob.glob1('examples', f'pred_fls_{audio_name[:-4]}*.txt')

        for i in range(0,len(fls)):
            fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
            fl[:, :, 0:2] = -fl[:, :, 0:2]
            fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

            if (ADD_NAIVE_EYE):
                fl = util.add_naive_eye(fl)

            # additional smooth
            fl = fl.reshape((-1, 204))
            fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
            fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
            fl = fl.reshape((-1, 68, 3))

            ''' STEP 6: Imag2image translation '''
            model = Image_translation_block(settings, single_test=True)
            with torch.no_grad():
                model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=image_name.split('.')[0])
                print('finish image2image gen')
            os.remove(os.path.join('examples', fls[i]))

        if(os.path.isfile(f'examples/{audio_name}')):
            os.remove(f'examples/{audio_name}')
        if(os.path.isfile(f'examples/{audio_name[:-4]}_av.mp4')):
            os.remove(f'examples/{audio_name[:-4]}_av.mp4')
        
        print(f"examples/{image_name.split('.')[0]}_pred_fls_{audio_name.split('.')[0]}_audio_embed.mp4")
        return f"examples/{image_name.split('.')[0]}_pred_fls_{audio_name.split('.')[0]}_audio_embed.mp4"

# audio = scipy.io.wavfile.read('examples/test_audio/M6_04_16k.wav')
# image = cv2.imread('examples/aya.jpg')
# print(end2vid.run(audio, image, 'test.wav', 'ayaaa.jpg'))
