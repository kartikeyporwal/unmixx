from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

#process single wave file
output_wav = myClearVoice(input_path='/home/kar/projects/unmixx/input/5Kn32cIWPSY_mono_16khz.wav', online_write=False)
myClearVoice.write(output_wav, output_path='output/output_5Kn32cIWPSY_mono_16khz_MossFormer2_SE_48K.wav')
