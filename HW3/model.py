import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from constants import F_MAX, F_MIN, HOP_SIZE, N_FFT, N_MELS, SAMPLE_RATE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_SIZE,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            normalized=False)

    def forward(self, audio):
        # Alignment correction to match with piano roll.
        # When they convert the input into frames,
        # pretty_midi.get_piano_roll uses `ceil`,
        # but torchaudio.transforms.melspectrogram uses `round`.
        padded_audio = nn.functional.pad(audio, (HOP_SIZE // 2, 0), 'constant')
        mel = self.melspectrogram(padded_audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        mel = torch.log(torch.clamp(mel, min=1e-9))
        return mel


class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size, 1 channel, frames, input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5))

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class Transcriber(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc = nn.Linear(fc_unit, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel)  # (B, T, C)
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)  # (B, T, C)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out


class Transcriber_RNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        self.cnn_unit = cnn_unit
        self.fc_unit = fc_unit

        self.frame_lstm = nn.LSTM(input_size= 229, hidden_size= 88, num_layers= 2, batch_first= True, bidirectional= True)
        # [direction * 2 (bidirection)]
        self.frame_fc = nn.Linear(88 * 2, 88)

        self.onset_lstm = nn.LSTM(input_size= 229, hidden_size= 88, num_layers= 2, batch_first= True, bidirectional= True)
        self.onset_fc = nn.Linear(88 * 2, 88)

    def forward(self, audio):
        # TODO: Question 1

        mel = self.melspectrogram(audio)
        # print(mel.shape)

        # [num_layers * num_bidirections, input_size, output_size] => hidden and cell
        frame_h = torch.zeros(2*2, mel.size(0), 88)
        frame_c = torch.zeros(2*2, mel.size(0), 88)
        onset_h = torch.zeros(2*2, mel.size(0), 88)
        onset_c = torch.zeros(2*2, mel.size(0), 88)

        frame_out, _ = self.frame_lstm(mel, (frame_h, frame_c))
        frame_out = self.frame_fc(frame_out)

        onset_out, _ = self.onset_lstm(mel, (onset_h, onset_c))
        onset_out = self.onset_fc(onset_out)

        return frame_out, onset_out


class Transcriber_CRNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_lstm = nn.LSTM(input_size= fc_unit, hidden_size= 88, num_layers= 2, bidirectional= True, batch_first= True)
        self.frame_fc = nn.Linear(88*2, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, bidirectional= True, batch_first= True)
        self.onset_fc = nn.Linear(88*2, 88)

    def forward(self, audio):
        # TODO: Question 2

        mel = self.melspectrogram(audio)
        # print(mel.shape)

        # [num_layers * num_bidirections, input_size, output_size] => hidden and cell
        frame_h = torch.zeros(2 * 2, mel.size(0), 88)
        frame_c = torch.zeros(2 * 2, mel.size(0), 88)
        onset_h = torch.zeros(2 * 2, mel.size(0), 88)
        onset_c = torch.zeros(2 * 2, mel.size(0), 88)

        # [B x T x C]
        frame_out = self.frame_conv_stack(mel)
        # print(frame_out.shape)

        frame_out, _ = self.frame_lstm(frame_out, (frame_h, frame_c))
        frame_out = self.frame_fc(frame_out)
        # print(frame_out.shape)

        onset_out = self.onset_conv_stack(mel)
        onset_out, _ = self.onset_lstm(onset_out, (onset_h, onset_c))
        onset_out = self.onset_fc(onset_out)

        return frame_out, onset_out


class Transcriber_ONF(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_lstm = nn.LSTM(input_size= 88 + 88, hidden_size=88, num_layers=2, batch_first=True,
                                  bidirectional=True)
        self.frame_fc2 = nn.Linear(88 * 2, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size= fc_unit, hidden_size=88, num_layers=2, batch_first=True,
                                  bidirectional=True)
        self.onset_fc = nn.Linear(88 * 2, 88)


    def forward(self, audio):
        # TODO: Question 3
        mel = self.melspectrogram(audio)

        frame_h = torch.zeros(2 * 2, mel.size(0), 88)
        frame_c = torch.zeros(2 * 2, mel.size(0), 88)
        onset_h = torch.zeros(2 * 2, mel.size(0), 88)
        onset_c = torch.zeros(2 * 2, mel.size(0), 88)

        # onset
        onset_out = self.onset_conv_stack(mel)  # (B x T x C)
        onset_out, _ = self.onset_lstm(onset_out, (onset_h, onset_c))
        onset_out = self.onset_fc(onset_out)
        # print(onset_out.shape)

        # frame
        frame_out = self.frame_conv_stack(mel)  # (B x T x C)
        frame_out = self.frame_fc1(frame_out)

        # concat the onset output and frame
        onset_out_copy = onset_out.clone()  # copy it on new variable
        onset_out_copy = onset_out_copy.detach()
        frame_out = torch.cat((frame_out, onset_out_copy), 2)  # concatenate

        frame_out, _ = self.frame_lstm(frame_out, (frame_h, frame_c))
        frame_out = self.frame_fc2(frame_out)
        # print(frame_out.shape)

        return frame_out, onset_out
