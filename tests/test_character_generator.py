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
    BODY_RANGES,
    FACE_REGION_PARAMS,
    FACIAL_FEATURES,
    CharacterGenerator,
)


class TestCharacterGenerator:
    def setup_method(self):
        self.system = CharacterGenerator()

    def test_case_01(self):
        # 相同种子生成完全相同的角色（可复现）
        c1 = self.system.generate_character(seed=42)
        c2 = self.system.generate_character(seed=42)
        assert c1.to_dict() == c2.to_dict()
        assert c1.seed == 42

    def test_case_02(self):
        # 不同种子生成不同角色；不传种子时也能随机生成
        c1 = self.system.generate_character(seed=1)
        c2 = self.system.generate_character(seed=2)
        assert c1.to_dict() != c2.to_dict()
        r1 = self.system.generate_character()
        r2 = self.system.generate_character()
        assert r1.to_dict() != r2.to_dict()

    def test_case_03(self):
        # 种子可跨实例复现角色
        other = CharacterGenerator()
        c1 = self.system.generate_character(seed=2024, name="hero")
        c2 = other.generate_character(seed=2024, name="hero")
        assert c1.to_dict() == c2.to_dict()

    def test_case_04(self):
        # 生成的角色包含完整的面部区域、特征与体型维度，且取值合法
        c = self.system.generate_character(seed=7)
        assert set(c.face.keys()) == set(FACE_REGION_PARAMS.keys())
        for region, params in FACE_REGION_PARAMS.items():
            assert set(c.face[region].keys()) == set(params)
            for value in c.face[region].values():
                assert 0.0 <= value <= 1.0
        assert set(c.features.keys()) == set(FACIAL_FEATURES)
        for dim, (low, high) in BODY_RANGES.items():
            assert low <= c.body[dim] <= high

    def test_case_05(self):
        # 捏脸：可按眼睛/鼻子/嘴巴等区域分别调整
        c = self.system.generate_character(seed=10)
        assert "eyes" in self.system.face_regions()
        assert "nose" in self.system.face_regions()
        assert "mouth" in self.system.face_regions()
        region = self.system.face_customization(c, "eyes", {"size": 0.9, "tilt": 0.2})
        assert region["size"] == 0.9
        assert region["tilt"] == 0.2
        assert c.face["eyes"]["size"] == 0.9
        # 调整眼睛不影响鼻子
        nose_before = dict(c.face["nose"])
        self.system.face_customization(c, "eyes", {"size": 0.1})
        assert c.face["nose"] == nose_before

    def test_case_06(self):
        # 捏脸：精细控制，支持极小步进微调并夹取到 [0,1]
        c = self.system.generate_character(seed=11)
        self.system.face_customization(c, "mouth", {"width": 0.5})
        self.system.face_customization(c, "mouth", {"width": 0.5 + 0.01})
        assert c.face["mouth"]["width"] == pytest.approx(0.51, abs=1e-4)
        region = self.system.face_customization(c, "mouth", {"width": 1.5})
        assert region["width"] == 1.0
        region = self.system.face_customization(c, "mouth", {"width": -0.3})
        assert region["width"] == 0.0

    def test_case_07(self):
        # 捏脸：未知区域或参数报错
        c = self.system.generate_character(seed=12)
        with pytest.raises(ValueError):
            self.system.face_customization(c, "forehead", {"size": 0.5})
        with pytest.raises(ValueError):
            self.system.face_customization(c, "eyes", {"not_a_param": 0.5})

    def test_case_08(self):
        # 面部特征：眼睛大小、鼻子高度、嘴巴宽度等参数可精细调整
        c = self.system.generate_character(seed=13)
        features = self.system.facial_features(c)
        for name in ("eye_size", "nose_height", "mouth_width"):
            assert name in features
        self.system.facial_features(c, {"eye_size": 0.8, "nose_height": 0.3, "mouth_width": 0.6})
        assert c.features["eye_size"]["left"] == 0.8
        assert c.features["nose_height"]["left"] == 0.3
        assert c.features["mouth_width"]["left"] == 0.6
        with pytest.raises(ValueError):
            self.system.facial_features(c, {"horn_length": 0.5})

    def test_case_09(self):
        # 面部特征：默认对称，可分别调整左右制造不对称，也可恢复对称
        c = self.system.generate_character(seed=14)
        assert self.system.facial_symmetry(c) == 1.0
        self.system.set_facial_feature(c, "eye_size", 0.9, side="left")
        self.system.set_facial_feature(c, "eye_size", 0.2, side="right")
        assert c.features["eye_size"]["left"] == 0.9
        assert c.features["eye_size"]["right"] == 0.2
        assert self.system.facial_symmetry(c) < 1.0
        self.system.facial_features(c, symmetry=1.0)
        assert c.features["eye_size"]["left"] == c.features["eye_size"]["right"]
        assert self.system.facial_symmetry(c) == 1.0

    def test_case_10(self):
        # 体型参数：不止身高体重，还有肩宽、腰围、腿长、臂长等维度
        c = self.system.generate_character(seed=15)
        for dim in ("shoulder_width", "waist", "leg_length", "arm_length"):
            assert dim in c.body
        result = self.system.body_parameters(
            c, {"shoulder_width": 55.0, "waist": 70.0, "leg_length": 95.0, "arm_length": 75.0}
        )
        assert result["dimensions"]["shoulder_width"] == 55.0
        assert result["dimensions"]["waist"] == 70.0
        assert result["dimensions"]["leg_length"] == 95.0
        assert result["dimensions"]["arm_length"] == 75.0
        # 超出合法范围会被夹取
        low, high = BODY_RANGES["height"]
        assert self.system.body_parameters(c, {"height": 999.0})["dimensions"]["height"] == high
        assert self.system.body_parameters(c, {"height": 1.0})["dimensions"]["height"] == low

    def test_case_11(self):
        # 体型参数：可按比例调整身体，比例与维度保持联动
        c = self.system.generate_character(seed=16)
        c.body["height"] = 180.0
        result = self.system.body_parameters(c, proportions={"leg_to_height": 0.5})
        assert c.body["leg_length"] == pytest.approx(90.0, abs=0.01)
        assert result["proportions"]["leg_to_height"] == pytest.approx(0.5, abs=0.01)
        with pytest.raises(ValueError):
            self.system.body_parameters(c, proportions={"wingspan": 1.0})

    def test_case_12(self):
        # 角色预设：可保存并加载角色，加载结果与保存时一致
        c = self.system.generate_character(seed=17, name="存档角色")
        self.system.face_customization(c, "nose", {"height": 0.77})
        self.system.save_preset("我的角色", c, category="realistic")
        loaded = self.system.load_preset("我的角色")
        assert loaded.to_dict() == c.to_dict()
        assert loaded.face["nose"]["height"] == 0.77
        with pytest.raises(KeyError):
            self.system.load_preset("不存在")

    def test_case_13(self):
        # 角色预设：可导出分享编码，其他玩家导入后得到相同角色
        c = self.system.generate_character(seed=18, name="分享角色")
        self.system.save_preset("分享", c, category="fantasy")
        code = self.system.share_preset("分享")
        assert isinstance(code, str) and len(code) > 0
        other = CharacterGenerator()
        imported = other.import_preset(code)
        assert imported.to_dict() == c.to_dict()
        assert "分享" in other.list_presets()
        with pytest.raises(ValueError):
            other.import_preset("!!!invalid-code!!!")

    def test_case_14(self):
        # 角色预设：可按风格分类与筛选
        gen = self.system
        gen.save_preset("写实A", gen.generate_character(seed=19), category="realistic")
        gen.save_preset("写实B", gen.generate_character(seed=20), category="realistic")
        gen.save_preset("卡通A", gen.generate_character(seed=21), category="cartoon")
        assert gen.list_presets() == ["写实A", "写实B", "卡通A"]
        assert gen.list_presets(category="realistic") == ["写实A", "写实B"]
        assert gen.list_presets(category="cartoon") == ["卡通A"]
        assert gen.preset_categories() == ["cartoon", "realistic"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
