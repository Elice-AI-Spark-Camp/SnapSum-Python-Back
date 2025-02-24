from enum import Enum

class VoiceName(str, Enum):
    male_1 = "ko-KR-Standard-C"  # 남성 음성 1
    male_2 = "ko-KR-Standard-D"  # 남성 음성 2
    female_1 = "ko-KR-Standard-A"  # 여성 음성 1
    female_2 = "ko-KR-Standard-B"  # 여성 음성 2
