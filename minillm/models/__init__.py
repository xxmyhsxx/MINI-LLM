"""模型自动注册。"""

from minillm.models.registry import register
from minillm.models.qwen2 import Qwen2ForCausalLM
from minillm.models.qwen3 import Qwen3ForCausalLM
from minillm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# 注册支持的模型
register("qwen2", Qwen2ForCausalLM)
register("qwen3", Qwen3ForCausalLM)
register("qwen2_5_vl", Qwen2_5_VLForConditionalGeneration)
