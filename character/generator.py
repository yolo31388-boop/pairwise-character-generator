"""
游戏内程序化角色生成与捏脸系统 - 核心模块

功能：
- generate_character(): 程序化随机生成角色，支持种子复现
- face_customization(): 按面部区域（眼睛/鼻子/嘴巴等）精细捏脸
- facial_features(): 面部特征参数化调整，支持左右对称控制
- body_parameters(): 体型维度（肩宽/腰围/腿长/臂长等）与身体比例调整
- character_preset(): 角色预设的保存/加载/分享/分类
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import base64
import copy
import hashlib
import json
import random


@dataclass
class Config:
    pass


# 面部区域及其可调参数（取值范围均为 [0.0, 1.0]）
FACE_REGION_PARAMS: Dict[str, tuple] = {
    "eyes": ("size", "spacing", "height", "tilt", "openness"),
    "nose": ("height", "width", "bridge", "tip_size"),
    "mouth": ("width", "lip_thickness", "corner_height", "fullness"),
    "jaw": ("width", "length", "chin_prominence"),
    "eyebrows": ("thickness", "arch", "height"),
    "cheeks": ("fullness", "height", "bone_prominence"),
    "ears": ("size", "protrusion"),
}

# 面部特征（左右成对，可分别调整以制造不对称）
FACIAL_FEATURES: tuple = (
    "eye_size", "eye_spacing", "nose_height", "nose_width",
    "mouth_width", "lip_thickness", "brow_height",
    "cheek_fullness", "jaw_width", "ear_size",
)

# 体型维度及其合法取值范围
BODY_RANGES: Dict[str, tuple] = {
    "height": (140.0, 210.0),
    "weight": (40.0, 150.0),
    "shoulder_width": (30.0, 60.0),
    "waist": (50.0, 130.0),
    "hip_width": (30.0, 60.0),
    "leg_length": (60.0, 110.0),
    "arm_length": (50.0, 90.0),
    "torso_length": (40.0, 70.0),
    "neck_length": (8.0, 18.0),
}

# 身体比例：比例名 -> (分子维度, 分母维度)
BODY_PROPORTIONS: Dict[str, tuple] = {
    "leg_to_height": ("leg_length", "height"),
    "arm_to_height": ("arm_length", "height"),
    "shoulder_to_hip": ("shoulder_width", "hip_width"),
    "torso_to_height": ("torso_length", "height"),
}

GENDERS = ("male", "female", "neutral")
STYLES = ("default", "realistic", "cartoon", "fantasy", "sci-fi")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class Character:
    id: str
    name: str
    seed: int
    gender: str
    style: str
    face: Dict[str, Dict[str, float]]
    features: Dict[str, Dict[str, float]]
    body: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Character":
        return cls(
            id=data["id"],
            name=data["name"],
            seed=data["seed"],
            gender=data["gender"],
            style=data["style"],
            face={k: dict(v) for k, v in data["face"].items()},
            features={k: dict(v) for k, v in data["features"].items()},
            body=dict(data["body"]),
        )


class CharacterGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._presets: Dict[str, Dict] = {}

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._presets = {}

    # ------------------------------------------------------------------
    # 角色生成（随机 + 种子复现）
    # ------------------------------------------------------------------
    def generate_character(
        self,
        seed: Optional[int] = None,
        name: str = "",
        gender: Optional[str] = None,
        style: str = "default",
    ) -> Character:
        """程序化随机生成一个角色。相同种子生成完全相同的角色。"""
        if seed is None:
            seed = random.SystemRandom().randrange(0, 2 ** 31)
        if style not in STYLES:
            raise ValueError(f"未知风格: {style}")
        rng = random.Random(seed)

        if gender is None:
            gender = rng.choice(GENDERS)
        if gender not in GENDERS:
            raise ValueError(f"未知性别: {gender}")

        face = {
            region: {param: round(rng.random(), 4) for param in params}
            for region, params in FACE_REGION_PARAMS.items()
        }
        features = {
            name: {"left": round(rng.random(), 4), "right": round(rng.random(), 4)}
            for name in FACIAL_FEATURES
        }
        # 默认面部保持对称
        for pair in features.values():
            pair["right"] = pair["left"]

        body = {
            dim: round(rng.uniform(low, high), 2)
            for dim, (low, high) in BODY_RANGES.items()
        }

        char_id = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()[:12]
        character = Character(
            id=f"char_{char_id}",
            name=name or f"角色_{char_id[:6]}",
            seed=seed,
            gender=gender,
            style=style,
            face=face,
            features=features,
            body=body,
        )
        self._history.append(("generate", seed))
        return character

    # ------------------------------------------------------------------
    # 捏脸：面部区域 + 精细控制
    # ------------------------------------------------------------------
    def face_regions(self) -> List[str]:
        """返回所有可调整的面部区域。"""
        return list(FACE_REGION_PARAMS.keys())

    def face_customization(
        self,
        character: Character,
        region: str,
        adjustments: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """对指定面部区域做精细调整，参数取值 [0.0, 1.0]，支持极小步进微调。"""
        if region not in FACE_REGION_PARAMS:
            raise ValueError(f"未知面部区域: {region}")
        updates = dict(adjustments or {})
        updates.update(kwargs)
        target = character.face[region]
        for param, value in updates.items():
            if param not in FACE_REGION_PARAMS[region]:
                raise ValueError(f"区域 {region} 没有参数: {param}")
            if not isinstance(value, (int, float)):
                raise ValueError(f"参数值必须是数字: {param}={value!r}")
            target[param] = round(_clamp(value, 0.0, 1.0), 4)
        self._history.append(("face_customization", region))
        return dict(target)

    # ------------------------------------------------------------------
    # 面部特征：参数化 + 左右对称
    # ------------------------------------------------------------------
    def facial_features(
        self,
        character: Character,
        updates: Optional[Dict[str, float]] = None,
        symmetry: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """读取/修改面部特征参数；symmetry ∈ [0,1] 控制左右脸对称程度。"""
        if updates:
            for name, value in updates.items():
                if isinstance(value, dict):
                    for side in ("left", "right"):
                        if side in value:
                            self.set_facial_feature(character, name, value[side], side=side)
                else:
                    self.set_facial_feature(character, name, value, side="both")
        if symmetry is not None:
            self.apply_symmetry(character, symmetry)
        return {k: dict(v) for k, v in character.features.items()}

    def set_facial_feature(
        self, character: Character, name: str, value: float, side: str = "both"
    ) -> None:
        """精细设置单个面部特征，可只调整左侧或右侧以制造不对称。"""
        if name not in FACIAL_FEATURES:
            raise ValueError(f"未知面部特征: {name}")
        if side not in ("left", "right", "both"):
            raise ValueError(f"side 必须是 left/right/both: {side}")
        clamped = round(_clamp(value, 0.0, 1.0), 4)
        pair = character.features[name]
        if side in ("left", "both"):
            pair["left"] = clamped
        if side in ("right", "both"):
            pair["right"] = clamped
        self._history.append(("set_facial_feature", name))

    def apply_symmetry(self, character: Character, symmetry: float) -> None:
        """按对称度 [0,1] 将左右特征向均值靠拢，1.0 表示完全对称。"""
        factor = _clamp(symmetry, 0.0, 1.0)
        for pair in character.features.values():
            mean = (pair["left"] + pair["right"]) / 2.0
            pair["left"] = round(pair["left"] + (mean - pair["left"]) * factor, 4)
            pair["right"] = round(pair["right"] + (mean - pair["right"]) * factor, 4)

    def facial_symmetry(self, character: Character) -> float:
        """返回当前面部对称度（1.0 为完全对称）。"""
        diffs = [abs(p["left"] - p["right"]) for p in character.features.values()]
        return round(1.0 - sum(diffs) / len(diffs), 4)

    # ------------------------------------------------------------------
    # 体型参数：维度 + 比例
    # ------------------------------------------------------------------
    def body_dimensions(self) -> List[str]:
        return list(BODY_RANGES.keys())

    def body_parameters(
        self,
        character: Character,
        dimensions: Optional[Dict[str, float]] = None,
        proportions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """读取/修改体型维度（自动夹取到合法范围）与身体比例。"""
        if dimensions:
            for dim, value in dimensions.items():
                self.set_body_dimension(character, dim, value)
        if proportions:
            for name, ratio in proportions.items():
                self.set_body_proportion(character, name, ratio)
        return {
            "dimensions": dict(character.body),
            "proportions": self.get_body_proportions(character),
        }

    def set_body_dimension(self, character: Character, dim: str, value: float) -> None:
        if dim not in BODY_RANGES:
            raise ValueError(f"未知体型维度: {dim}")
        low, high = BODY_RANGES[dim]
        character.body[dim] = round(_clamp(value, low, high), 2)
        self._history.append(("set_body_dimension", dim))

    def get_body_proportions(self, character: Character) -> Dict[str, float]:
        return {
            name: round(character.body[num] / character.body[den], 4)
            for name, (num, den) in BODY_PROPORTIONS.items()
        }

    def set_body_proportion(self, character: Character, name: str, ratio: float) -> None:
        """按比例调整身体（如 leg_to_height=0.55 会同步修改腿长）。"""
        if name not in BODY_PROPORTIONS:
            raise ValueError(f"未知身体比例: {name}")
        num, den = BODY_PROPORTIONS[name]
        self.set_body_dimension(character, num, character.body[den] * float(ratio))

    # ------------------------------------------------------------------
    # 角色预设：保存/加载/分享/分类
    # ------------------------------------------------------------------
    def save_preset(self, name: str, character: Character, category: str = "default") -> Dict:
        if not name:
            raise ValueError("预设名称不能为空")
        preset = {
            "name": name,
            "category": category or "default",
            "character": character.to_dict(),
        }
        self._presets[name] = preset
        self._history.append(("save_preset", name))
        return copy.deepcopy(preset)

    def load_preset(self, name: str) -> Character:
        if name not in self._presets:
            raise KeyError(f"预设不存在: {name}")
        return Character.from_dict(copy.deepcopy(self._presets[name]["character"]))

    def delete_preset(self, name: str) -> None:
        if name not in self._presets:
            raise KeyError(f"预设不存在: {name}")
        del self._presets[name]

    def list_presets(self, category: Optional[str] = None) -> List[str]:
        return sorted(
            name
            for name, preset in self._presets.items()
            if category is None or preset["category"] == category
        )

    def preset_categories(self) -> List[str]:
        return sorted({preset["category"] for preset in self._presets.values()})

    def share_preset(self, name: str) -> str:
        """将预设导出为可分享的字符串编码，可发送给其他玩家。"""
        if name not in self._presets:
            raise KeyError(f"预设不存在: {name}")
        payload = json.dumps(self._presets[name], ensure_ascii=False, sort_keys=True)
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")

    def import_preset(
        self, code: str, name: Optional[str] = None, category: Optional[str] = None
    ) -> Character:
        """导入其他玩家分享的预设编码，并保存到本地预设库。"""
        try:
            payload = json.loads(base64.urlsafe_b64decode(code.encode("ascii")).decode("utf-8"))
            character = Character.from_dict(payload["character"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"无效的分享编码: {exc}") from exc
        preset_name = name or payload.get("name") or character.name
        self.save_preset(preset_name, character, category or payload.get("category", "default"))
        self._history.append(("import_preset", preset_name))
        return character
