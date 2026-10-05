"""
游戏内程序化角色生成与捏脸系统 - 核心模块

功能：
- generate_character(): 程序化随机生成角色，支持 seed 复现
- face_customization(): 按面部区域（眼/鼻/嘴等）精细捏脸
- facial_features(): 面部特征参数，支持左右对称/不对称调整
- body_parameters(): 多维度体型参数与身体比例
- character_preset(): 角色预设的保存/加载/分享码/分类
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import base64
import copy
import json
import random
import time


@dataclass
class Config:
    pass


# 面部区域 -> 该区域可精细调整的特征
FACE_REGION_FEATURES: Dict[str, tuple] = {
    "eyes": ("eye_size", "eye_spacing", "eye_height", "eye_tilt"),
    "nose": ("nose_height", "nose_width", "bridge_height", "tip_size"),
    "mouth": ("mouth_width", "lip_thickness", "mouth_height", "corner_lift"),
    "jaw": ("jaw_width", "chin_length", "chin_width"),
    "brows": ("brow_height", "brow_thickness", "brow_arch"),
}

FACE_REGIONS: tuple = tuple(FACE_REGION_FEATURES)

FEATURE_REGION: Dict[str, str] = {
    feature: region
    for region, features in FACE_REGION_FEATURES.items()
    for feature in features
}

FACIAL_FEATURES: tuple = tuple(FEATURE_REGION)

FACE_MIN, FACE_MAX = 0.0, 1.0

# 体型维度（厘米/公斤）及其合法取值范围
BODY_DIMENSION_RANGES: Dict[str, tuple] = {
    "height": (140.0, 210.0),
    "weight": (40.0, 150.0),
    "shoulder_width": (30.0, 60.0),
    "waist": (50.0, 130.0),
    "hip_width": (25.0, 55.0),
    "leg_length": (60.0, 110.0),
    "arm_length": (50.0, 90.0),
    "neck_length": (8.0, 18.0),
    "torso_length": (40.0, 70.0),
}

BODY_DIMENSIONS: tuple = tuple(BODY_DIMENSION_RANGES)

PRESET_CATEGORIES = ("realistic", "anime", "fantasy", "sci-fi", "custom")


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class CharacterGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state: Dict = {}
        self._history: List = []
        self._presets: Dict[str, Dict] = {}

    def update(self, dt: float):
        # 游戏主循环钩子（保留原接口）
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._presets = {}

    # ------------------------------------------------------------------
    # 角色程序化生成
    # ------------------------------------------------------------------
    def generate_character(self, seed: Optional[int] = None,
                           category: str = "custom") -> Dict:
        """随机生成一个角色；相同 seed 必定生成相同角色。"""
        if category not in PRESET_CATEGORIES:
            raise ValueError(f"未知角色分类: {category!r}")

        actual_seed = seed if seed is not None else random.randrange(2 ** 63)
        rng = random.Random(actual_seed)

        face = {}
        for region, features in FACE_REGION_FEATURES.items():
            face[region] = {
                feature: {
                    "left": round(rng.uniform(0.25, 0.75), 4),
                    "right": round(rng.uniform(0.25, 0.75), 4),
                }
                for feature in features
            }

        body = {
            dim: round(rng.uniform(low, high), 2)
            for dim, (low, high) in BODY_DIMENSION_RANGES.items()
        }

        character = {
            "id": f"{actual_seed:016x}-{rng.getrandbits(32):08x}",
            "seed": actual_seed,
            "category": category,
            "face": face,
            "body": body,
            "proportions": self._compute_proportions(body),
        }
        self._history.append(character["id"])
        return character

    # ------------------------------------------------------------------
    # 捏脸：按面部区域精细调整
    # ------------------------------------------------------------------
    def face_customization(self, character: Dict, region: str,
                           adjustments: Dict[str, float],
                           relative: bool = False) -> Dict:
        """对某个面部区域（eyes/nose/mouth/...）的特征做精细调整。

        adjustments: {特征名: 目标值或增量}，取值范围 0.0~1.0
        relative=True 时在当前值基础上叠加增量，否则直接设置目标值。
        """
        if region not in FACE_REGION_FEATURES:
            raise ValueError(
                f"未知面部区域: {region!r}，可选: {', '.join(FACE_REGIONS)}"
            )

        region_face = character["face"][region]
        for feature, value in adjustments.items():
            if feature not in FACE_REGION_FEATURES[region]:
                raise ValueError(
                    f"区域 {region!r} 不包含特征 {feature!r}"
                )
            for side in ("left", "right"):
                current = region_face[feature][side]
                target = current + float(value) if relative else float(value)
                region_face[feature][side] = round(
                    _clamp(target, FACE_MIN, FACE_MAX), 4
                )
        return character

    # ------------------------------------------------------------------
    # 面部特征参数（可指定左/右，支持对称与不对称）
    # ------------------------------------------------------------------
    def facial_features(self, character: Dict, feature: str, value: float,
                        side: str = "both") -> Dict:
        """设置单个面部特征参数，side 为 left/right/both。"""
        if feature not in FEATURE_REGION:
            raise ValueError(
                f"未知面部特征: {feature!r}，可选: {', '.join(FACIAL_FEATURES)}"
            )
        if side not in ("left", "right", "both"):
            raise ValueError(f"side 必须是 left/right/both，实际: {side!r}")

        region = FEATURE_REGION[feature]
        target = round(_clamp(float(value), FACE_MIN, FACE_MAX), 4)
        sides = ("left", "right") if side == "both" else (side,)
        for target_side in sides:
            character["face"][region][feature][target_side] = target
        return character

    def get_facial_feature(self, character: Dict, feature: str) -> Dict[str, float]:
        region = self._require_feature(feature)
        return dict(character["face"][region][feature])

    def facial_asymmetry(self, character: Dict, feature: str) -> float:
        """返回某特征左右脸差异（0 表示完全对称）。"""
        region = self._require_feature(feature)
        value = character["face"][region][feature]
        return round(abs(value["left"] - value["right"]), 4)

    def _require_feature(self, feature: str) -> str:
        if feature not in FEATURE_REGION:
            raise ValueError(f"未知面部特征: {feature!r}")
        return FEATURE_REGION[feature]

    # ------------------------------------------------------------------
    # 体型参数：多维度 + 比例
    # ------------------------------------------------------------------
    def body_parameters(self, character: Dict, **dimensions: float) -> Dict:
        """设置一个或多个体型维度，并重新计算身体比例。"""
        body = character["body"]
        for dim, value in dimensions.items():
            if dim not in BODY_DIMENSION_RANGES:
                raise ValueError(
                    f"未知体型维度: {dim!r}，可选: {', '.join(BODY_DIMENSIONS)}"
                )
            low, high = BODY_DIMENSION_RANGES[dim]
            body[dim] = round(_clamp(float(value), low, high), 2)
        character["proportions"] = self._compute_proportions(body)
        return character

    @staticmethod
    def _compute_proportions(body: Dict[str, float]) -> Dict[str, float]:
        height = body["height"]
        return {
            "leg_ratio": round(body["leg_length"] / height, 4),
            "arm_ratio": round(body["arm_length"] / height, 4),
            "torso_ratio": round(body["torso_length"] / height, 4),
            "neck_ratio": round(body["neck_length"] / height, 4),
            "shoulder_to_waist": round(
                body["shoulder_width"] / body["waist"], 4
            ),
            "shoulder_to_hip": round(
                body["shoulder_width"] / body["hip_width"], 4
            ),
            "bmi": round(body["weight"] / ((height / 100.0) ** 2), 2),
        }

    # ------------------------------------------------------------------
    # 角色预设：保存 / 加载 / 分享 / 分类
    # ------------------------------------------------------------------
    def character_preset(self, action: str, name: Optional[str] = None,
                         character: Optional[Dict] = None,
                         category: Optional[str] = None,
                         share_code: Optional[str] = None):
        if action == "save":
            return self._save_preset(name, character, category)
        if action == "load":
            return self._load_preset(name)
        if action == "delete":
            if name is None:
                raise ValueError("删除预设需要提供 name")
            self._presets.pop(name, None)
            return True
        if action == "share":
            return self._share_preset(name)
        if action == "import":
            return self._import_preset(share_code, name)
        if action == "list":
            return self._list_presets(category)
        raise ValueError(
            f"未知预设操作: {action!r}，支持 "
            "save/load/delete/share/import/list"
        )

    def _save_preset(self, name, character, category):
        if not name:
            raise ValueError("保存预设需要提供 name")
        if character is None:
            raise ValueError("保存预设需要提供 character")
        preset_category = category or character.get("category", "custom")
        if preset_category not in PRESET_CATEGORIES:
            raise ValueError(f"未知角色分类: {preset_category!r}")
        self._presets[name] = {
            "name": name,
            "category": preset_category,
            "character": copy.deepcopy(character),
            "created_at": time.time(),
        }
        return copy.deepcopy(self._presets[name])

    def _load_preset(self, name):
        if name not in self._presets:
            raise KeyError(f"预设不存在: {name!r}")
        return copy.deepcopy(self._presets[name]["character"])

    def _share_preset(self, name):
        if name not in self._presets:
            raise KeyError(f"预设不存在: {name!r}")
        preset = self._presets[name]
        payload = json.dumps(
            {
                "name": preset["name"],
                "category": preset["category"],
                "character": preset["character"],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")

    def _import_preset(self, share_code, name=None):
        if not share_code:
            raise ValueError("导入预设需要提供 share_code")
        raw = json.loads(base64.urlsafe_b64decode(share_code.encode("ascii")))
        character = copy.deepcopy(raw["character"])
        if name:
            self._save_preset(name, character, raw.get("category"))
        return character

    def _list_presets(self, category=None):
        if category is not None and category not in PRESET_CATEGORIES:
            raise ValueError(f"未知角色分类: {category!r}")
        result = [
            {"name": preset["name"], "category": preset["category"]}
            for preset in self._presets.values()
            if category is None or preset["category"] == category
        ]
        return sorted(result, key=lambda item: item["name"])
