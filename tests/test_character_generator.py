"""
游戏内程序化角色生成与捏脸系统 - 验收测试
运行方式：python -m pytest tests/test_character_generator.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from character.generator import (
    BODY_DIMENSIONS,
    FACE_REGIONS,
    CharacterGenerator,
)


class TestCharacterGenerator:
    def setup_method(self):
        self.system = CharacterGenerator()

    # ---- 角色生成：随机 + 种子 ----

    def test_case_01(self):
        """生成是程序化的：连续生成的角色互不相同，不是固定预设"""
        chars = [self.system.generate_character() for _ in range(5)]
        ids = {c["id"] for c in chars}
        assert len(ids) == 5
        faces = {repr(c["face"]) for c in chars}
        bodies = {repr(c["body"]) for c in chars}
        assert len(faces) > 1
        assert len(bodies) > 1

    def test_case_02(self):
        """相同种子生成完全相同的角色（可复现）"""
        first = self.system.generate_character(seed=42)
        second = self.system.generate_character(seed=42)
        assert first == second
        assert first["seed"] == 42

    def test_case_03(self):
        """不同种子生成不同角色"""
        first = self.system.generate_character(seed=1)
        second = self.system.generate_character(seed=2)
        assert first["id"] != second["id"]
        assert first["face"] != second["face"] or first["body"] != second["body"]

    # ---- 捏脸：面部区域 + 精细控制 ----

    def test_case_04(self):
        """捏脸支持按面部区域调整（眼睛/鼻子/嘴巴）"""
        char = self.system.generate_character(seed=7)
        for region in ("eyes", "nose", "mouth"):
            assert region in char["face"]
        self.system.face_customization(char, "eyes", {"eye_size": 0.9})
        self.system.face_customization(char, "nose", {"nose_height": 0.3})
        self.system.face_customization(char, "mouth", {"mouth_width": 0.6})
        assert char["face"]["eyes"]["eye_size"]["left"] == 0.9
        assert char["face"]["nose"]["nose_height"]["right"] == 0.3
        assert char["face"]["mouth"]["mouth_width"]["left"] == 0.6

    def test_case_05(self):
        """调整一个区域不影响其他区域"""
        char = self.system.generate_character(seed=11)
        mouth_before = repr(char["face"]["mouth"])
        nose_before = repr(char["face"]["nose"])
        self.system.face_customization(char, "eyes", {"eye_size": 0.1})
        assert repr(char["face"]["mouth"]) == mouth_before
        assert repr(char["face"]["nose"]) == nose_before
        with pytest.raises(ValueError):
            self.system.face_customization(char, "horns", {"eye_size": 0.5})

    def test_case_06(self):
        """捏脸支持精细微调（小增量叠加并夹取到合法范围）"""
        char = self.system.generate_character(seed=13)
        self.system.face_customization(char, "eyes", {"eye_size": 0.5})
        self.system.face_customization(
            char, "eyes", {"eye_size": 0.001}, relative=True
        )
        assert char["face"]["eyes"]["eye_size"]["left"] == 0.501
        self.system.face_customization(char, "eyes", {"eye_size": 5.0})
        assert char["face"]["eyes"]["eye_size"]["left"] == 1.0

    # ---- 面部特征：参数 + 对称 ----

    def test_case_07(self):
        """面部特征有参数：眼睛大小、鼻子高度、嘴巴宽度等"""
        char = self.system.generate_character(seed=17)
        self.system.facial_features(char, "eye_size", 0.8)
        self.system.facial_features(char, "nose_height", 0.4)
        self.system.facial_features(char, "mouth_width", 0.65)
        assert self.system.get_facial_feature(char, "eye_size") == {
            "left": 0.8, "right": 0.8,
        }
        assert self.system.get_facial_feature(char, "nose_height")["left"] == 0.4
        assert self.system.get_facial_feature(char, "mouth_width")["right"] == 0.65

    def test_case_08(self):
        """面部特征支持左右分别调整（可对称也可不对称）"""
        char = self.system.generate_character(seed=19)
        self.system.facial_features(char, "eye_size", 0.8, side="left")
        self.system.facial_features(char, "eye_size", 0.3, side="right")
        assert self.system.facial_asymmetry(char, "eye_size") == 0.5
        self.system.facial_features(char, "eye_size", 0.6, side="both")
        assert self.system.facial_asymmetry(char, "eye_size") == 0.0

    # ---- 体型参数：维度 + 比例 ----

    def test_case_09(self):
        """体型有多个维度：肩宽、腰围、腿长、臂长等"""
        char = self.system.generate_character(seed=23)
        for dim in ("shoulder_width", "waist", "leg_length", "arm_length"):
            assert dim in BODY_DIMENSIONS
        self.system.body_parameters(
            char,
            shoulder_width=52.0,
            waist=80.0,
            leg_length=95.0,
            arm_length=70.0,
        )
        assert char["body"]["shoulder_width"] == 52.0
        assert char["body"]["waist"] == 80.0
        assert char["body"]["leg_length"] == 95.0
        assert char["body"]["arm_length"] == 70.0

    def test_case_10(self):
        """体型参数会驱动身体比例，且超出范围会被夹取"""
        char = self.system.generate_character(seed=29)
        self.system.body_parameters(
            char, height=180.0, leg_length=90.0, shoulder_width=54.0, waist=90.0
        )
        proportions = char["proportions"]
        assert proportions["leg_ratio"] == 0.5
        assert proportions["shoulder_to_waist"] == 0.6
        self.system.body_parameters(char, height=999.0)
        assert char["body"]["height"] == 210.0
        assert proportions is not char["proportions"]

    # ---- 角色预设：保存/加载/分享/分类 ----

    def test_case_11(self):
        """预设可以保存并完整加载"""
        char = self.system.generate_character(seed=31)
        self.system.facial_features(char, "eye_size", 0.77)
        self.system.character_preset("save", name="hero", character=char)
        loaded = self.system.character_preset("load", name="hero")
        assert loaded == char
        loaded["body"]["height"] = 140.0
        again = self.system.character_preset("load", name="hero")
        assert again["body"]["height"] == char["body"]["height"]

    def test_case_12(self):
        """预设可以通过分享码分享给其他玩家"""
        char = self.system.generate_character(seed=37)
        self.system.character_preset(
            "save", name="shared", character=char, category="anime"
        )
        code = self.system.character_preset("share", name="shared")
        assert isinstance(code, str) and len(code) > 0

        other_player = CharacterGenerator()
        imported = other_player.character_preset("import", share_code=code)
        assert imported == char
        other_player.character_preset(
            "import", share_code=code, name="received"
        )
        assert other_player.character_preset("load", name="received") == char

    def test_case_13(self):
        """预设支持按风格分类与筛选"""
        self.system.character_preset(
            "save", name="a", character=self.system.generate_character(seed=41),
            category="anime",
        )
        self.system.character_preset(
            "save", name="b", character=self.system.generate_character(seed=43),
            category="realistic",
        )
        self.system.character_preset(
            "save", name="c", character=self.system.generate_character(seed=47),
            category="anime",
        )
        all_presets = self.system.character_preset("list")
        assert [p["name"] for p in all_presets] == ["a", "b", "c"]
        anime = self.system.character_preset("list", category="anime")
        assert [p["name"] for p in anime] == ["a", "c"]
        assert all(p["category"] == "anime" for p in anime)

    def test_case_14(self):
        """完整流程：种子生成 -> 捏脸 -> 特征 -> 体型 -> 预设分享复现"""
        char = self.system.generate_character(seed=100, category="fantasy")
        self.system.face_customization(char, "mouth", {"mouth_width": 0.72})
        self.system.facial_features(char, "nose_height", 0.55, side="left")
        self.system.body_parameters(char, height=185.0, shoulder_width=55.0)
        self.system.character_preset("save", name="final", character=char)

        code = self.system.character_preset("share", name="final")
        friend = CharacterGenerator()
        restored = friend.character_preset("import", share_code=code)

        assert restored["seed"] == 100
        assert restored["face"]["mouth"]["mouth_width"]["left"] == 0.72
        assert friend.facial_asymmetry(restored, "nose_height") > 0
        assert restored["body"]["height"] == 185.0
        assert restored["proportions"]["leg_ratio"] == round(
            restored["body"]["leg_length"] / 185.0, 4
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
