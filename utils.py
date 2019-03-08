import torch
import collections

# Transcription part
class str2LabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self.ignore_case = ignore_case
        if self.ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'

        # 0 is reserved for 'blank' required by wrap_ctc
        self.encode_dict = {letter: idx + 1 for idx, letter in enumerate(alphabet)}
        self.decode_dict = ' ' + alphabet


    def encode(self, text):
        """
        support batch or single str.

        :param text: text(str or list of str): text to encode
        :return: torch.IntTensor [length_0 + length_1 + ... + length{n - 1}]: encoded text
                 torch.IntTensor [n]: length of each text in batch
        """
        if isinstance(text, str):
            text = [
                self.encode_dict[word.lower() if self.ignore_case
                else word] for word in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_code, length):
        """
        Decode encoded texts back into strs.
        :param text_code: encoded text torch.
        :param length: length of each text in batch
        :return: text (str of list of str)
        """
        if length.numel() == 1: # single str
            length = length[0]
            assert text_code.numel() == length, \
                'text with length: {} does not match declared length: {}'.format(text_code.numel(), length)
            word_list = []
            for idx in range(length):
                if text_code[idx] != 0 and (not(idx > 0 and text_code[idx] == text_code[idx - 1])):
                    word_list.append(self.alphabet[text_code[idx] - 1])
                return ''.join(word_list)
        else:
            # batch mode
            assert text_code.numel() == length.sum(), \
                'text with length: {} does not match declared length: {}'.format(text_code.numel(), length.sum())
            texts = []
            idx = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(text_code[idx:idx + l], torch.IntTensor([l]))
                )
                idx += l
            return texts