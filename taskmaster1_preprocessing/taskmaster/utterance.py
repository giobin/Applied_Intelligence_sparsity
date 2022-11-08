import copy
import math

USER_TAG_START = "<user>"
USER_TAG_END = "<enduser>"

AGENT_TAG_START = "<agent>"
AGENT_TAG_END = "<endagent>"


class Utterance:
    '''
    Manage the conversations loaded from the Taskmaster-1 dataset.
    '''

    def __init__(self, utterances, index):
        '''
        :param utterances: Conversation extracted from Taskmaster.
        :param index: ID of the conversation.
        '''
        self.id = index
        self.utterances = utterances

    def getSentencesWithContext(self):
        '''
        Generate from the utterance multiple sentences with context.
        :return: list of [sentence with context, response]
        '''
        sentences = []

        for i in range(0, len(self.utterances), 2):
            utterance = self.utterances[i]
            utterance_next = self.getNext(i)
            if i == 0:
                sentence = []
                sentence.append(get_user_sentence(utterance['text']))
                sentence.append(get_agent_sentence(utterance_next))
                sentences.append(sentence)
            else:
                sentence = copy.copy(sentences[len(sentences) - 1])
                sentence[0] = sentence[0] + sentence[1] + get_user_sentence(utterance['text'])
                sentence[1] = get_agent_sentence(utterance_next)
                sentences.append(sentence)
        assert len(sentences) == math.ceil(len(self.utterances) / 2)

        return sentences

    def getSentences(self):
        '''
        Generate from the utterance multiple sentences without context.
        :return: list of [sentence, response]
        '''
        sentences = []

        for i in range(0, len(self.utterances), 2):
            utterance = self.utterances[i]
            utterance_next = self.getNext(i)
            sentence = []
            sentence.append(get_user_sentence(utterance['text']))
            sentence.append(get_agent_sentence(utterance_next))
            sentences.append(sentence)

        return sentences

    def getNext(self, i):
        if i + 1 < len(self.utterances):
            return self.utterances[i + 1]['text']
        else:
            return "<no_response>"


def get_user_sentence(str):
    return USER_TAG_START + str + USER_TAG_END


def get_agent_sentence(str):
    return AGENT_TAG_START + str + AGENT_TAG_END
