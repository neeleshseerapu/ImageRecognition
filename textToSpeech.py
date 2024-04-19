import pyttsx4  # had to also download py3-tts module


class TTS:
    """A class that converts text to sound"""
    def __init__(self, rate=200, volume=1.0, voice_num=7):
        self.engine = pyttsx4.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_num].id)

    def speak(self, string):
        """Speaks using engine settings"""
        self.engine.say(string)
        self.engine.runAndWait()
        self.engine.stop()

    def set_rate(self, rate):
        """Changes the speed at which the voice talks"""
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """Changes the volume of the voice"""
        self.engine.setProperty('volume', volume)
