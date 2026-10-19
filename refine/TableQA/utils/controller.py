"""
阶段一：最小可行 Controller（与现有代码兼容）

该模块实现了 Controller 框架，可以在不修改现有流程的前提下，
通过包装现有函数调用来添加 Controller 层。

功能：
- ControllerState: 状态管理
- ControllerAction: 动作空间定义
- Decision: 决策结果
- MinimalController: 决策逻辑
- ActionExecutor: 动作执行器
- controller_main_loop: 主循环
"""

import re
import json
import copy
import os
import pickle
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from contextlib import nullcontext

# 调试模式开关
# DEBUG = os.environ.get("CONTROLLER_DEBUG", "false").lower() == "true"
DEBUG = True

# 统一的错误树 JSON 路径
CRITIC_TREE_JSON = "critic/TableQA/tools/few_shot_critic.json"


# ==================== 1. ControllerState 数据类 ====================

@dataclass
class ControllerState:
    """
    Controller 状态管理类
    
    属性:
        sample: 从现有 sample 提取的完整数据
        question: 问题描述
        chain: 推理链
        conclusion: 结论 ("[Correct]" or "[Incorrect]")
        iteration: 当前迭代次数
        last_action: 上一次执行的动作
        error_route: 错误路由
        retrieved_blueprints: 检索到的蓝图
        diagnosis: 诊断结果
        action_history: 历史动作记录
    """
    # 从现有 sample 提取
    sample: Dict[str, Any]
    question: str
    chain: List[Dict]
    conclusion: str  # "[Correct]" or "[Incorrect]"
    
    # Controller 管理的状态
    iteration: int = 0
    last_action: str = "INIT"
    error_route: Optional[str] = None
    retrieved_blueprints: Optional[List[str]] = None
    retrieval_info: str = ""  # 阶段2：存储LLM决策时需要的检索信息
    # 新增：存储预检索的 few-shot 示例，避免重复检索
    retrieved_few_shot_examples: Optional[List[Any]] = None
    diagnosis: Optional[Dict] = None
    
    # 历史摘要
    action_history: List[str] = field(default_factory=list)
    
    @property
    def is_correct(self) -> bool:
        """判断当前状态是否正确"""
        return "[Correct]" in self.conclusion


# ==================== 2. ControllerAction 枚举 ====================

class ControllerAction(Enum):
    """
    简化动作空间（与现有流程对应）
    
    STOP: 停止（对应现有流程的结束）
    EXECUTE_TREE: 执行 tree_exec_one_sample
    DIAGNOSE_BP: 诊断（blueprint 模式）
    DIAGNOSE_FS: 诊断（few-shot 模式）
    REFINE_CHAIN: 精炼推理链
    REFINE_QUERY: 精炼最终查询
    UPDATE_TREE: 更新错误树
    """
    STOP = "STOP"
    EXECUTE_TREE = "EXECUTE_TREE"
    DIAGNOSE_BP = "DIAGNOSE_BP"
    DIAGNOSE_FS = "DIAGNOSE_FS"
    REFINE_CHAIN = "REFINE_CHAIN"
    REFINE_QUERY = "REFINE_QUERY"
    UPDATE_TREE = "UPDATE_TREE"


# ==================== 3. Decision 数据类 ====================

@dataclass
class Decision:
    """
    决策结果
    
    属性:
        action: 选择的动作
        reason: 决策原因
        confidence: 置信度 (0.0-1.0)
        source: 决策来源 ("rule" | "llm")
    """
    action: ControllerAction
    reason: str
    confidence: float
    source: str  # "rule" | "llm"


# ==================== 4. MinimalController 类 ====================

class MinimalController:
    """
    最小 Controller、
    
    决策逻辑：
    1. 规则层：如果正确，停止
    2. 规则层：达到最大迭代次数，停止
    3. LLM 层：决定下一步动作（可选）
    """
    
    def __init__(self, llm, llm_options, max_iterations: int = 2, retriever=None):
        """
        初始化 Controller
        
        Args:
            llm: 语言模型实例
            max_iterations: 最大迭代次数
            retriever: RetrieverAgent 实例（避免重复创建）
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.llm_options = llm_options
        # 修复1：统一使用传入的 RetrieverAgent 实例，避免缓存不一致
        self.retriever = retriever
    
    def decide(self, state: ControllerState) -> Decision:
        """
        基于当前状态决定下一步动作
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        
        # 规则层：如果正确，停止
        if state.is_correct:
            return Decision(
                action=ControllerAction.STOP,
                reason="Already correct",
                confidence=1.0,
                source="rule"
            )
        
        # 规则层：达到最大迭代次数，停止
        if state.iteration >= self.max_iterations:
            return Decision(
                action=ControllerAction.STOP,
                reason=f"Max iterations ({self.max_iterations}) reached",
                confidence=0.8,
                source="rule"
            )
        
        # LLM 层：决定下一步动作
        return self._llm_decide(state)
    
    def _llm_decide(self, state: ControllerState) -> Decision:
        """
        使用 LLM 决策（通用版）
        
        根据当前状态决定下一步动作：
        - EXECUTE_TREE 后：决定 DIAGNOSE_BP 还是 DIAGNOSE_FS
        - DIAGNOSE 后：决定 REFINE_CHAIN 还是 REFINE_QUERY
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        try:
            # 根据上一步动作决定 LLM 决策类型
            if state.last_action == ControllerAction.EXECUTE_TREE.value or state.last_action == "INIT":
                # EXECUTE_TREE 后：决定 DIAGNOSE_BP 还是 DIAGNOSE_FS
                return self._llm_decide_diagnose(state)
            elif state.last_action in [ControllerAction.DIAGNOSE_BP.value, ControllerAction.DIAGNOSE_FS.value]:
                # DIAGNOSE 后：决定 REFINE_CHAIN 还是 REFINE_QUERY
                return self._llm_decide_refine(state)
            else:
                # 默认使用规则决策
                return self._default_decision(state)
        except Exception as e:
            # LLM 决策失败时，直接停止流程
            print(f"LLM decision failed: {e}, stopping flow")
            # return self._default_decision(state)
            return Decision(
                action=ControllerAction.STOP,
                reason=f"LLM decision failed: {e}",
                confidence=0.0,
                source="llm"
            )
    
    def _default_decision(self, state: ControllerState) -> Decision:
        """
        默认决策（模拟现有流程 _judge_critic_refine_with_cache_mp_core）
        
        决策流程：
        1. 初始Judge判断 → 如果正确直接返回
        2. EXECUTE_TREE → 获取 error_route
        3. DIAGNOSE_BP (iteration=1) / DIAGNOSE_FS (iteration=2)
        4. 根据 incorrect_step 决定 REFINE_CHAIN 或 REFINE_QUERY
        5. Judge → 如果正确更新树并返回
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        # 根据当前状态和迭代次数决定下一步动作
        if state.iteration == 0:
            # 第一次迭代：执行树分类获取 error_route
            return Decision(
                action=ControllerAction.EXECUTE_TREE,
                reason="First iteration: execute tree to get error_route",
                confidence=0.9,
                source="rule"
            )
        elif state.iteration == 1:
            # 第二次迭代：使用 blueprint 模式诊断
            if state.last_action == ControllerAction.EXECUTE_TREE:
                return Decision(
                    action=ControllerAction.DIAGNOSE_BP,
                    reason="Second iteration: diagnose with blueprint_only=True",
                    confidence=0.9,
                    source="rule"
                )
            elif state.last_action in [ControllerAction.DIAGNOSE_BP, ControllerAction.DIAGNOSE_FS]:
                # 诊断后，根据 incorrect_step 决定精炼类型
                from refine.TableQA.utils.extract_step import return_incorrect_max_step
                incorrect_step, max_step = return_incorrect_max_step(state.sample)
                if incorrect_step and incorrect_step != max_step:
                    return Decision(
                        action=ControllerAction.REFINE_CHAIN,
                        reason="Error in reasoning chain (incorrect_step != max_step), refine chain",
                        confidence=0.9,
                        source="rule"
                    )
                else:
                    return Decision(
                        action=ControllerAction.REFINE_QUERY,
                        reason="Error in final query (incorrect_step == max_step), refine query",
                        confidence=0.9,
                        source="rule"
                    )
        elif state.iteration == 2:
            # 第三次迭代：使用 full few-shot 模式诊断
            if state.last_action in [ControllerAction.EXECUTE_TREE, ControllerAction.DIAGNOSE_BP]:
                return Decision(
                    action=ControllerAction.DIAGNOSE_FS,
                    reason="Third iteration: diagnose with blueprint_only=False (few-shot)",
                    confidence=0.8,
                    source="rule"
                )
            elif state.last_action in [ControllerAction.DIAGNOSE_FS, ControllerAction.REFINE_CHAIN, ControllerAction.REFINE_QUERY]:
                # 诊断/精炼后，根据 incorrect_step 决定精炼类型
                from refine.TableQA.utils.extract_step import return_incorrect_max_step
                incorrect_step, max_step = return_incorrect_max_step(state.sample)
                if incorrect_step and incorrect_step != max_step:
                    return Decision(
                        action=ControllerAction.REFINE_CHAIN,
                        reason="Error in reasoning chain, refine chain",
                        confidence=0.8,
                        source="rule"
                    )
                else:
                    return Decision(
                        action=ControllerAction.REFINE_QUERY,
                        reason="Error in final query, refine query",
                        confidence=0.8,
                        source="rule"
                    )
        
        # 默认停止
        return Decision(
            action=ControllerAction.STOP,
            reason="No more actions available",
            confidence=0.5,
            source="rule"
        )
    
    def _llm_decide_refine(self, state: ControllerState) -> Decision:
        """
        使用 LLM 决定下一步是 REFINE_CHAIN 还是 REFINE_QUERY
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        # 获取诊断信息
        from refine.TableQA.utils.extract_step import return_incorrect_max_step
        incorrect_step, max_step = return_incorrect_max_step(state.sample)
        
        # 阶段2：使用_llm_decide_diagnose中存储的retrieval_info
        # 不再重复调用RetrieverAgent
        retrieval_info = getattr(state, 'retrieval_info', '')
        if retrieval_info:
            # 将 "Blueprint" 改为 "Error blueprint" 以适应refine上下文
            retrieval_info = retrieval_info.replace("- Blueprint available:", "- Error blueprint:")
        
        # 构建 LLM 决策 prompt
        prompt = f"""Current state:
- Iteration: {state.iteration}
- Last action: {state.last_action}
- Error route: {state.error_route or "N/A"}
- Diagnosis conclusion: {state.diagnosis.get("conclusion", "N/A") if state.diagnosis else "N/A"}
- Incorrect step: {incorrect_step}
- Max step: {max_step}
- Retrieval_info: {retrieval_info}

Based on the diagnosis, decide next action:
- REFINE_CHAIN: If the error is in the reasoning chain (incorrect_step != max_step)
- REFINE_QUERY: If the error is only in the final query (incorrect_step == max_step)

Output JSON: {{"action": "...", "reason": "...", "confidence": 0.0-1.0}}"""
        
        try:
            response = self.llm.generate(
                prompt,
                options=self.llm_options
            )
            decision_data = json.loads(response)
            
            # 验证决策
            action_str = decision_data.get("action", "")
            if "CHAIN" in action_str.upper():
                action = ControllerAction.REFINE_CHAIN
            elif "QUERY" in action_str.upper():
                action = ControllerAction.REFINE_QUERY
            else:
                # 默认根据 incorrect_step 决定
                if incorrect_step and incorrect_step != max_step:
                    action = ControllerAction.REFINE_CHAIN
                else:
                    action = ControllerAction.REFINE_QUERY
            
            return Decision(
                action=action,
                reason=decision_data.get("reason", ""),
                confidence=decision_data.get("confidence", 0.5),
                source="llm"
            )
        except Exception as e:
            # Fallback: 使用规则
            print(f"LLM refine decision failed: {e}, using rule-based decision")
            if incorrect_step and incorrect_step != max_step:
                return Decision(
                    action=ControllerAction.REFINE_CHAIN,
                    reason="Error in reasoning chain (incorrect_step != max_step), refine chain",
                    confidence=0.9,
                    source="rule"
                )
            else:
                return Decision(
                    action=ControllerAction.REFINE_QUERY,
                    reason="Error in final query (incorrect_step == max_step), refine query",
                    confidence=0.9,
                    source="rule"
                )
    
    def _llm_decide_diagnose(self, state: ControllerState) -> Decision:
        """
        使用 LLM 决定下一步是 DIAGNOSE_BP 还是 DIAGNOSE_FS
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        # 构建 LLM 决策 prompt
        # 阶段2：使用 RetrieverAgent 获取检索结果增强上下文
        # 注意：将retrieval_info存储到state中，传递给_llm_decide_refine使用
        # 优化：一次检索，多处使用，避免重复检索
        retrieval_info = ""
        if state.error_route and not state.retrieval_info:
            try:
                # 修复1：使用传入的 retriever 实例（避免重复创建）
                if self.retriever is not None:
                    result = self.retriever.retrieve_by_route(state.error_route)
                else:
                    # 向后兼容：如果没有传入 retriever，创建临时实例
                    from agents import RetrieverAgent
                    retriever = RetrieverAgent(memory_path=CRITIC_TREE_JSON)
                    result = retriever.retrieve_by_route(state.error_route)
                
                # 存储完整结果到 state，避免后续重复检索
                state.retrieved_blueprints = result.get('blueprint')
                state.retrieved_few_shot_examples = result.get('few_shot_examples', [])
                
                # 修复2：修正键名错误，使用 'few_shot_examples' 而不是 'content'/'contents'
                if result.get('blueprint'):
                    retrieval_info = f"\n- Blueprint available: {result['blueprint']}..."
                if result.get('few_shot_examples'):
                    retrieval_info += f"\n- Few-shot examples available: {len(result['few_shot_examples'])} examples"
                # 存储到state中，传递给_llm_decide_refine
                state.retrieval_info = retrieval_info
                
                if DEBUG:
                    few_shot_count = len(state.retrieved_few_shot_examples) if state.retrieved_few_shot_examples else 0
                    print(f"[DEBUG RETRIEVER] Stored retrieved data in state:")
                    print(f"  - Blueprint: {state.retrieved_blueprints is not None}")
                    print(f"  - Few-shot count: {few_shot_count}")
            except Exception as e:
                print(f"[ERROR] RetrieverAgent error in _llm_decide_diagnose: {e}", flush=True)
                print(f"Full traceback: {traceback.format_exc()}", flush=True)
        
        prompt = f"""Current state:
- Iteration: {state.iteration}
- Last action: {state.last_action}
- Error route: {state.error_route or "N/A"}
- Has been diagnosed with blueprint: {ControllerAction.DIAGNOSE_BP.value in state.action_history}{retrieval_info}

Decide next action:
- DIAGNOSE_BP: Use blueprint-only mode (faster, less context)
- DIAGNOSE_FS: Use few-shot mode (slower, more context with examples)

Use DIAGNOSE_BP if you haven't tried blueprint mode yet.
Use DIAGNOSE_FS if blueprint mode failed or you need more context.

Output JSON: {{"action": "...", "reason": "...", "confidence": 0.0-1.0}}"""
        
        try:
            response = self.llm.generate(
                prompt,
                options=self.llm_options
            )
            decision_data = json.loads(response)
            
            # 验证决策
            action_str = decision_data.get("action", "")
            if "BP" in action_str.upper() and "FS" not in action_str.upper():
                action = ControllerAction.DIAGNOSE_BP
            elif "FS" in action_str.upper():
                action = ControllerAction.DIAGNOSE_FS
            else:
                # 默认：如果之前没有 DIAGNOSE_BP，使用 DIAGNOSE_BP
                if ControllerAction.DIAGNOSE_BP.value not in state.action_history:
                    action = ControllerAction.DIAGNOSE_BP
                else:
                    action = ControllerAction.DIAGNOSE_FS
            
            return Decision(
                action=action,
                reason=decision_data.get("reason", ""),
                confidence=decision_data.get("confidence", 0.5),
                source="llm"
            )
        except Exception as e:
            # Fallback: 使用规则
            print(f"LLM diagnose decision failed: {e}, using rule-based decision")
            if ControllerAction.DIAGNOSE_BP.value not in state.action_history:
                return Decision(
                    action=ControllerAction.DIAGNOSE_BP,
                    reason="First diagnose: use blueprint mode",
                    confidence=0.9,
                    source="rule"
                )
            else:
                return Decision(
                    action=ControllerAction.DIAGNOSE_FS,
                    reason="Blueprint mode done: use few-shot mode",
                    confidence=0.8,
                    source="rule"
                )


# ==================== 阶段3：EnhancedController 类 ====================

class EnhancedController(MinimalController):
    """
    增强版 Controller
    
    在 MinimalController 的基础上添加：
    - 更智能的决策逻辑
    - 增强版 prompt 构建（包含更多上下文）
    - 决策验证机制
    
    目标：添加更智能的决策逻辑
    """
    
    def __init__(self, llm, llm_options, max_iterations: int = 2, retriever=None):
        """
        初始化增强版 Controller
        
        Args:
            llm: 语言模型实例
            llm_options: 模型选项
            max_iterations: 最大迭代次数
            retriever: RetrieverAgent 实例
        """
        super().__init__(llm, llm_options, max_iterations, retriever)
    
    def _llm_decide(self, state: ControllerState) -> Decision:
        """
        使用增强版 LLM 决策
        
        与 MinimalController 的区别：
        - 使用增强版 prompt（包含更多上下文）
        - 添加决策验证机制
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        try:
            # 根据上一步动作决定 LLM 决策类型
            if state.last_action == ControllerAction.EXECUTE_TREE.value or state.last_action == "INIT":
                # EXECUTE_TREE 后：决定 DIAGNOSE_BP 还是 DIAGNOSE_FS
                return self._llm_decide_diagnose_enhanced(state)
            elif state.last_action in [ControllerAction.DIAGNOSE_BP.value, ControllerAction.DIAGNOSE_FS.value]:
                # DIAGNOSE 后：决定 REFINE_CHAIN 还是 REFINE_QUERY
                return self._llm_decide_refine_enhanced(state)
            else:
                # 默认使用规则决策
                return self._default_decision(state)
        except Exception as e:
            # LLM 决策失败时，直接停止流程
            print(f"[EnhancedController] LLM decision failed: {e}, stopping flow")
            return Decision(
                action=ControllerAction.STOP,
                reason=f"LLM decision failed: {e}",
                confidence=0.0,
                source="llm"
            )
    
    def _llm_decide_diagnose_enhanced(self, state: ControllerState) -> Decision:
        """
        使用增强版 LLM 决定下一步是 DIAGNOSE_BP 还是 DIAGNOSE_FS
        
        与 MinimalController 的区别：
        - 使用 _build_enhanced_prompt 构建更丰富的上下文
        - 使用 _validate_decision 验证决策的合理性
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        # 构建增强版 prompt
        prompt = self._build_enhanced_prompt(state, decision_type="diagnose")
        
        try:
            response = self.llm.generate(
                prompt,
                options=self.llm_options
            )
            decision_data = json.loads(response)
            
            # 验证决策
            return self._validate_decision(decision_data, state, decision_type="diagnose")
            
        except Exception as e:
            # Fallback: 使用规则
            print(f"[EnhancedController] LLM diagnose decision failed: {e}, using rule-based decision")
            if ControllerAction.DIAGNOSE_BP.value not in state.action_history:
                return Decision(
                    action=ControllerAction.DIAGNOSE_BP,
                    reason="First diagnose: use blueprint mode",
                    confidence=0.9,
                    source="rule"
                )
            else:
                return Decision(
                    action=ControllerAction.DIAGNOSE_FS,
                    reason="Blueprint mode done: use few-shot mode",
                    confidence=0.8,
                    source="rule"
                )
    
    def _llm_decide_refine_enhanced(self, state: ControllerState) -> Decision:
        """
        使用增强版 LLM 决定下一步是 REFINE_CHAIN 还是 REFINE_QUERY
        
        与 MinimalController 的区别：
        - 使用 _build_enhanced_prompt 构建更丰富的上下文
        - 使用 _validate_decision 验证决策的合理性
        
        Args:
            state: 当前状态
            
        Returns:
            Decision: 决策结果
        """
        # 构建增强版 prompt
        prompt = self._build_enhanced_prompt(state, decision_type="refine")
        
        try:
            response = self.llm.generate(
                prompt,
                options=self.llm_options
            )
            decision_data = json.loads(response)
            
            # 验证决策
            return self._validate_decision(decision_data, state, decision_type="refine")
            
        except Exception as e:
            # Fallback: 使用规则
            print(f"[EnhancedController] LLM refine decision failed: {e}, using rule-based decision")
            from refine.TableQA.utils.extract_step import return_incorrect_max_step
            incorrect_step, max_step = return_incorrect_max_step(state.sample)
            if incorrect_step and incorrect_step != max_step:
                return Decision(
                    action=ControllerAction.REFINE_CHAIN,
                    reason="Error in reasoning chain (incorrect_step != max_step), refine chain",
                    confidence=0.9,
                    source="rule"
                )
            else:
                return Decision(
                    action=ControllerAction.REFINE_QUERY,
                    reason="Error in final query (incorrect_step == max_step), refine query",
                    confidence=0.9,
                    source="rule"
                )
    
    def _build_enhanced_prompt(self, state: ControllerState, decision_type: str) -> str:
        """
        构建增强版 prompt
        
        与 MinimalController 的区别：
        - 包含更多上下文信息
        - 包含检索结果（blueprint 和 few-shot 示例）
        - 包含决策指南
        
        Args:
            state: 当前状态
            decision_type: 决策类型 ("diagnose" 或 "refine")
            
        Returns:
            str: 增强版 prompt
        """
        # 获取检索结果
        retrieval_info = ""
        if state.error_route and not state.retrieval_info:
            try:
                if self.retriever is not None:
                    result = self.retriever.retrieve_by_route(state.error_route)
                else:
                    from agents import RetrieverAgent
                    retriever = RetrieverAgent(memory_path=CRITIC_TREE_JSON)
                    result = retriever.retrieve_by_route(state.error_route)
                
                # 存储完整结果到 state
                state.retrieved_blueprints = result.get('blueprint')
                state.retrieved_few_shot_examples = result.get('few_shot_examples', [])
                
                if result.get('blueprint'):
                    retrieval_info = f"\n- Blueprint available: {result['blueprint']}..."
                if result.get('few_shot_examples'):
                    retrieval_info += f"\n- Few-shot examples available: {len(result['few_shot_examples'])} examples"
                state.retrieval_info = retrieval_info
                
                if DEBUG:
                    few_shot_count = len(state.retrieved_few_shot_examples) if state.retrieved_few_shot_examples else 0
                    print(f"[DEBUG ENHANCED] Stored retrieved data in state:")
                    print(f"  - Blueprint: {state.retrieved_blueprints is not None}")
                    print(f"  - Few-shot count: {few_shot_count}")
            except Exception as e:
                print(f"[ERROR] RetrieverAgent error in _build_enhanced_prompt: {e}", flush=True)
        
        # 使用已存储的检索信息
        retrieval_info = state.retrieval_info or ""
        
        # 获取诊断信息（仅用于 refine 决策）
        diagnosis_info = ""
        if decision_type == "refine" and state.diagnosis:
            try:
                from refine.TableQA.utils.extract_step import return_incorrect_max_step
                incorrect_step, max_step = return_incorrect_max_step(state.sample)
                diagnosis_info = f"""
- Diagnosis conclusion: {state.diagnosis.get("conclusion", "N/A")}
- Incorrect step: {incorrect_step}
- Max step: {max_step}"""
            except Exception as e:
                # 如果获取失败，只使用诊断结论
                diagnosis_info = f"""
- Diagnosis conclusion: {state.diagnosis.get("conclusion", "N/A")}"""
                if DEBUG:
                    print(f"[DEBUG ENHANCED] Failed to get incorrect/max_step: {e}")
            if retrieval_info:
                # 将 "Blueprint" 改为 "Error blueprint" 以适应refine上下文
                retrieval_info = retrieval_info.replace("- Blueprint available:", "- Error blueprint:")
        
        # 构建 prompt
        prompt = f"""You are a reasoning controller. Decide the next action.

Current State:
- Question: {state.question}
- Iteration: {state.iteration}
- Last action: {state.last_action}
- Error route: {state.error_route or "N/A"}
- Has been diagnosed with blueprint: {ControllerAction.DIAGNOSE_BP.value in state.action_history}{retrieval_info}{diagnosis_info}

"""
        
        if decision_type == "diagnose":
            prompt += """Available Actions:
- DIAGNOSE_BP: Diagnose error using blueprint only (faster, less context)
- DIAGNOSE_FS: Diagnose error using blueprint + few-shot examples (slower, more context)

Decision Guidelines:
1. First iteration after EXECUTE_TREE: Always start with DIAGNOSE_BP
2. If DIAGNOSE_BP failed or you need more context: Use DIAGNOSE_FS
3. Max iterations: {max_iterations}

Output JSON: {{"action": "...", "reason": "...", "confidence": 0.0-1.0}}""".format(max_iterations=self.max_iterations)
        
        elif decision_type == "refine":
            prompt += """Available Actions:
- REFINE_CHAIN: Refine reasoning chain from error step (if error is in reasoning chain)
- REFINE_QUERY: Refine only the final query (if error is only in final query)

Decision Guidelines:
1. If incorrect_step != max_step: Error is in reasoning chain, use REFINE_CHAIN
2. If incorrect_step == max_step: Error is only in final query, use REFINE_QUERY
3. Max iterations: {max_iterations}

Output JSON: {{"action": "...", "reason": "...", "confidence": 0.0-1.0}}""".format(max_iterations=self.max_iterations)
        
        return prompt
    
    def _validate_decision(
        self, 
        decision_data: Dict, 
        state: ControllerState,
        decision_type: str
    ) -> Decision:
        """
        验证 LLM 决策的合理性
        
        规则验证：
        - 第一次必须执行 tree（iteration=0, last_action=INIT）
        - 没有 error_route 不能诊断
        - 没有诊断不能精炼
        - 不正确不能更新树
        
        Args:
            decision_data: LLM 返回的决策数据
            state: 当前状态
            decision_type: 决策类型 ("diagnose" 或 "refine")
            
        Returns:
            Decision: 验证后的决策
        """
        action_str = decision_data.get("action", "")
        
        # 规则验证
        if decision_type == "diagnose":
            # 验证诊断决策
            if state.error_route is None and action_str.startswith("DIAGNOSE"):
                # 没有 error_route 不能诊断
                print(f"[EnhancedController] Cannot diagnose without error_route, forcing EXECUTE_TREE")
                decision_data["action"] = "EXECUTE_TREE"
                decision_data["reason"] = "Need error_route before diagnosis"
            
            elif ControllerAction.DIAGNOSE_BP.value in state.action_history and action_str == "DIAGNOSE_BP":
                # 已经尝试过 DIAGNOSE_BP，应该尝试 DIAGNOSE_FS
                print(f"[EnhancedController] Already tried DIAGNOSE_BP, forcing DIAGNOSE_FS")
                decision_data["action"] = "DIAGNOSE_FS"
                decision_data["reason"] = "Already tried blueprint mode, use few-shot mode"
        
        elif decision_type == "refine":
            # 验证精炼决策
            if state.diagnosis is None and action_str.startswith("REFINE"):
                # 没有诊断不能精炼
                print(f"[EnhancedController] Cannot refine without diagnosis, forcing DIAGNOSE_BP")
                decision_data["action"] = "DIAGNOSE_BP"
                decision_data["reason"] = "Need diagnosis before refinement"
            
            elif not state.is_correct and action_str == "UPDATE_TREE":
                # 不正确不能更新树
                print(f"[EnhancedController] Cannot update tree with incorrect answer, forcing STOP")
                decision_data["action"] = "STOP"
                decision_data["reason"] = "Only update tree with correct answers"
        
        # 转换为 ControllerAction
        action_str = decision_data.get("action", "")
        action = None
        
        if decision_type == "diagnose":
            if "BP" in action_str.upper() and "FS" not in action_str.upper():
                action = ControllerAction.DIAGNOSE_BP
            elif "FS" in action_str.upper():
                action = ControllerAction.DIAGNOSE_FS
            elif "TREE" in action_str.upper():
                action = ControllerAction.EXECUTE_TREE
            else:
                # 默认：如果之前没有 DIAGNOSE_BP，使用 DIAGNOSE_BP
                if ControllerAction.DIAGNOSE_BP.value not in state.action_history:
                    action = ControllerAction.DIAGNOSE_BP
                else:
                    action = ControllerAction.DIAGNOSE_FS
        
        elif decision_type == "refine":
            if "CHAIN" in action_str.upper():
                action = ControllerAction.REFINE_CHAIN
            elif "QUERY" in action_str.upper():
                action = ControllerAction.REFINE_QUERY
            else:
                # 默认根据 incorrect_step 决定
                from refine.TableQA.utils.extract_step import return_incorrect_max_step
                incorrect_step, max_step = return_incorrect_max_step(state.sample)
                if incorrect_step and incorrect_step != max_step:
                    action = ControllerAction.REFINE_CHAIN
                else:
                    action = ControllerAction.REFINE_QUERY
        
        # 如果 action 仍然是 None，使用默认决策
        if action is None:
            print(f"[EnhancedController] Failed to parse action: {action_str}, using default decision")
            return self._default_decision(state)
        
        return Decision(
            action=action,
            reason=decision_data.get("reason", ""),
            confidence=decision_data.get("confidence", 0.5),
            source="llm"
        )


# ==================== 5. ActionExecutor 类 ====================

class ActionExecutor:
    """
    动作执行器
    
    负责执行 Controller 决策的动作，并更新状态
    """
    
    def __init__(self, llm, llm_options, use_verifier=False, use_diff_critic=False, diff_critic_mode="diagnose_only"):
        """
        初始化执行器

        Args:
            llm: 语言模型实例
            llm_options: 模型选项
        """
        self.llm = llm
        self.llm_options = llm_options
        self.use_verifier = use_verifier
        self.use_diff_critic = use_diff_critic
        self.diff_critic_mode = diff_critic_mode
        
        # 初始化 RetrieverAgent（阶段2：独立检索器）
        from agents import RetrieverAgent
        self.retriever = RetrieverAgent(
            llm=llm,
            memory_path=CRITIC_TREE_JSON
        )
    
    def execute(self, state: ControllerState, decision: Decision) -> ControllerState:
        """
        执行决策并更新状态
        
        Args:
            state: 当前状态
            decision: 决策结果
            
        Returns:
            ControllerState: 更新后的状态
        """
        action = decision.action
        
        if action == ControllerAction.STOP:
            return state
        
        elif action == ControllerAction.EXECUTE_TREE:
            from critic.TableQA.tools import tree_exec_one_sample
            tree_sample = tree_exec_one_sample(
                state.sample, 
                llm=self.llm, 
                llm_options=self.llm_options
            )
            routes = re.findall(r'\((.*?)\)', tree_sample.get('tree', ''))
            # routes = re.findall(r'\((.*?)\)', tree_sample['tree'])
            state.error_route = routes[0] if routes else "random"
            state.sample = tree_sample
        
        elif action in [ControllerAction.DIAGNOSE_BP, ControllerAction.DIAGNOSE_FS]:
            from critic.TableQA.tools import critic_exec_one_sample
            blueprint_only = (action == ControllerAction.DIAGNOSE_BP)
            error_route = state.error_route or "random"

            # Diff-Critic: build diff bundle before Critic call
            diff_bundle = None
            if self.use_diff_critic:
                try:
                    from refine.TableQA.utils.diff_critic.diff_bundle import build_diff_bundle
                    diff_bundle = build_diff_bundle(state.sample)
                    state.sample["_diff_localization"] = {
                        "step_num": diff_bundle.localization.step_num,
                        "confidence": diff_bundle.localization.confidence,
                        "reason": diff_bundle.localization.reason,
                    }
                    if DEBUG:
                        loc = diff_bundle.localization
                        print(f"[DEBUG DIFF_CRITIC] Bundle built: step={loc.step_num}, conf={loc.confidence:.2f}, reason={loc.reason}")
                except Exception as exc:
                    print(f"[WARN] Diff bundle failed: {exc}", flush=True)
                    diff_bundle = None
            
            # 优化：检查 state 中是否已经有预检索的数据，避免重复检索
            # 更严格地检查：区分 None 和空列表
            pre_retrieved_few_shot = getattr(state, 'retrieved_few_shot_examples', None)
            
            # 只有在预检索数据存在且不为空时才使用
            # 如果预检索数据为 None，表示从未进行过检索，需要重新检索
            # 如果预检索数据为 []（空列表），表示检索过但结果为空，也需要重新检索
            if pre_retrieved_few_shot is not None and pre_retrieved_few_shot:
                if DEBUG:
                    shot_count = len(pre_retrieved_few_shot) if pre_retrieved_few_shot is not None else 0
                    print(f"[DEBUG RETRIEVER] Using pre-retrieved few-shot data: {shot_count} examples")
                # 使用预检索的数据，传递给 critic_exec_one_sample
                critic_sample = critic_exec_one_sample(
                    state.sample,
                    error_route,
                    llm=self.llm,
                    llm_options=self.llm_options,
                    blueprint_only=blueprint_only,
                    pre_retrieved_few_shot=pre_retrieved_few_shot,
                    use_verifier=self.use_verifier,
                    use_diff_critic=self.use_diff_critic,
                    diff_bundle=diff_bundle,
                )
            else:
                # 如果没有预检索数据，才调用 RetrieverAgent（向后兼容）
                if DEBUG:
                    print(f"[DEBUG RETRIEVER] No pre-retrieved data, fetching for route: {error_route}")
                try:
                    retrieval_result = self.retriever.retrieve_by_route(error_route)
                    state.retrieved_blueprints = retrieval_result.get('blueprint')
                    pre_retrieved_few_shot = retrieval_result.get('few_shot_examples', [])

                    if DEBUG:
                        print(f"[DEBUG RETRIEVER] Retrieved blueprint: {state.retrieved_blueprints is not None}")
                        print(f"[DEBUG RETRIEVER] Retrieved few_shot count: {len(pre_retrieved_few_shot)}")
                except Exception as e:
                    print(f"[ERROR] RetrieverAgent retrieve_by_route failed: {e}", flush=True)
                    print(f"Full traceback: {traceback.format_exc()}", flush=True)
                    pre_retrieved_few_shot = None

                # 执行 critic
                critic_sample = critic_exec_one_sample(
                    state.sample,
                    error_route,
                    llm=self.llm,
                    llm_options=self.llm_options,
                    blueprint_only=blueprint_only,
                    pre_retrieved_few_shot=pre_retrieved_few_shot,
                    use_verifier=self.use_verifier,
                    use_diff_critic=self.use_diff_critic,
                    diff_bundle=diff_bundle,
                )
            
            state.diagnosis = {
                "critique": critic_sample.get("critique", ""),
                "conclusion": critic_sample.get("conclusion", ""),
                "max_step": critic_sample.get("max_step", 0)
            }
            state.sample = critic_sample
        
        elif action == ControllerAction.REFINE_CHAIN:
            from refine.TableQA.utils.chain import dynamic_chain_exec_one_sample, get_table_info
            from refine.TableQA.operations.final_query import simple_query_cot_original
            from refine.TableQA.utils.extract_step import return_incorrect_max_step
            incorrect_step, max_step = return_incorrect_max_step(state.sample)
            if DEBUG:
                print(f"[DEBUG REFINE_CHAIN] Before dynamic_chain_exec_one_sample:")
                print(f"  - incorrect_step: {incorrect_step}")
                print(f"  - max_step: {max_step}")
                print(f"  - chain length: {len(state.sample.get('chain', []))}")
                if state.sample.get('chain'):
                    last_op = state.sample['chain'][-1]
                    print(f"  - last operation: {last_op.get('operation_name', 'N/A')}")
            refine_sample = dynamic_chain_exec_one_sample(
                state.sample,
                incorrect_step=incorrect_step,
                max_step=max_step,
                llm=self.llm,
                llm_options=self.llm_options,
                strategy="top"
            )
            if DEBUG:
                print(f"[DEBUG REFINE_CHAIN] After dynamic_chain_exec_one_sample:")
                print(f"  - chain length: {len(refine_sample.get('chain', []))}")
                if refine_sample.get('chain'):
                    last_op = refine_sample['chain'][-1]
                    print(f"  - last operation: {last_op.get('operation_name', 'N/A')}")
                    has_query = 'query' in last_op.get('operation_name', '').lower()
                    print(f"  - has query operation: {has_query}")
            
            # 修复：添加 query 操作
            # dynamic_chain_exec_one_sample 只生成中间操作，不包含 query 操作
            # 需要调用 simple_query_cot_original 添加最终的 query 操作
            table_info = get_table_info(
                refine_sample,
                skip_op=[],
                first_n_op=None,
            )
            if DEBUG:
                print(f"[DEBUG REFINE_CHAIN] Adding query operation...")
            refine_sample = simple_query_cot_original(
                refine_sample,
                table_info,
                self.llm,
                debug=DEBUG,
                use_demo=True,
                llm_options=self.llm.get_model_options(temperature=0, per_example_max_decode_steps=2048, per_example_top_p=1.0)
            )
            if DEBUG:
                print(f"[DEBUG REFINE_CHAIN] After adding query operation:")
                print(f"  - chain length: {len(refine_sample.get('chain', []))}")
                if refine_sample.get('chain'):
                    last_op = refine_sample['chain'][-1]
                    print(f"  - last operation: {last_op.get('operation_name', 'N/A')}")
            
            state.sample = refine_sample
        
        elif action == ControllerAction.REFINE_QUERY:
            from refine.TableQA.utils.chain import simple_query_with_critic, get_table_info
            table_info = get_table_info(
                state.sample,
                skip_op=[],
                first_n_op=None
            )
            refine_sample = simple_query_with_critic(
                state.sample,
                table_info,
                self.llm,
                llm_options=self.llm_options
            )
            state.sample = refine_sample
        
        elif action == ControllerAction.UPDATE_TREE:
            if frozen_memory:
                if DEBUG:
                    print("[DEBUG UPDATE_TREE] frozen_memory=True, skip update_error_tree")
            else:
                from critic.TableQA.tools import update_error_tree
                # 调试信息
                print(f"[DEBUG UPDATE_TREE] error_route: {state.error_route}")
                print(f"[DEBUG UPDATE_TREE] sample keys: {list(state.sample.keys())}")
                print(f"[DEBUG UPDATE_TREE] critique exists: {'critique' in state.sample}")
                print(f"[DEBUG UPDATE_TREE] conclusion: {state.sample.get('conclusion', 'N/A')}")
                
                # 使用 nullcontext() 替代 lock=None，支持上下文管理器协议
                update_error_tree(
                    state.sample,
                    state.error_route or "random",
                    error_tree_json=CRITIC_TREE_JSON,
                    llm=self.llm,
                    llm_options=self.llm_options,
                    lock=nullcontext(),
                    use_blueprint=True
                )
                print(f"[DEBUG UPDATE_TREE] Finished update_error_tree call")
        
        # 更新状态
        state.iteration += 1
        state.last_action = action.value
        state.action_history.append(action.value)
        
        # 更新 conclusion
        state.conclusion = state.sample.get("conclusion", "[Incorrect]")
        
        return state


# ==================== 6. load_clarifier_info 函数 ====================

def load_clarifier_info(sample: Dict[str, Any], cache_dir: Optional[str] = None, sample_idx: Optional[int] = None, thought_results_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 Clarifier 信息
    
    加载策略：
    1. 优先从 sample['clarifier'] 读取
    2. 如果 sample 中没有，尝试从 Thought 阶段的 clarifier 目录读取
    3. 如果都没有，尝试从独立缓存文件读取
    4. 如果都没有，输出警告并返回空字典
    
    Args:
        sample: 输入样本
        cache_dir: 缓存目录路径
        sample_idx: 样本索引
        thought_results_dir: Thought 阶段的结果目录路径（用于读取 clarifier 信息）
    
    Returns:
        Clarifier 信息字典
    """
    # 策略1: 从 sample 中读取
    if 'clarifier' in sample and bool(sample.get('clarifier')):
        return sample['clarifier']

    # 策略1.3: 从用户提供的 thought_results_dir 读取 clarifier
    if thought_results_dir and sample_idx is not None:
        try:
            thought_clarifier_dir = os.path.join(thought_results_dir, 'clarifier')
            clarifier_file = os.path.join(thought_clarifier_dir, f"case_dict_test-{sample_idx}.pkl")
            if os.path.exists(clarifier_file):
                with open(clarifier_file, 'rb') as f:
                    clarifier_data = pickle.load(f)
                    if DEBUG:
                        print(f"[DEBUG CLARIFIER] Loaded from {clarifier_file}")
                    return clarifier_data
        except Exception as e:
            print(f"[WARNING] Failed to load clarifier from thought_results_dir: {e}")

    # 策略1.5: 从 cache_dir 推断 Thought 阶段的 clarifier 目录读取
    # cache_dir 格式: results/refine_clarifier/{dataset}/{model}/{timestamp}/cache
    # thought_clarifier_dir 格式: results/thought/{dataset}/{model}/clarifier
    # clarifier 文件格式: case_dict_test-{sample_idx}.pkl
    if cache_dir and sample_idx is not None:
        try:
            # 从 Refine 阶段的 cache_dir 推断 Thought 阶段的 clarifier 目录
            # 例如: results/refine_clarifier/tabfact/qwen3:32b/20260328_145239/cache
            #       results/thought/tabfact/qwen3:32b/clarifier
            parts = cache_dir.split('/')
            # if DEBUG:
            #     print(f"[CLARIFIER DEBUG] cache_dir parts: {parts}")
            #     print(f"[CLARIFIER DEBUG] len(parts): {len(parts)}, parts[1]: {parts[1] if len(parts) > 1 else 'N/A'}")

            if len(parts) >= 5 and parts[1] == 'refine_clarifier':
                # 构建thought_clarifier_dir: results/thought/{dataset}/{model}/clarifier
                thought_clarifier_dir = os.path.join('results', 'thought', parts[2], parts[3], 'clarifier')
                clarifier_file = os.path.join(thought_clarifier_dir, f"case_dict_test-{sample_idx}.pkl")

                # if DEBUG:
                #     print(f"[CLARIFIER DEBUG] Inferred thought_clarifier_dir: {thought_clarifier_dir}")
                #     print(f"[CLARIFIER DEBUG] Looking for clarifier file: {clarifier_file}")
                #     print(f"[CLARIFIER DEBUG] File exists: {os.path.exists(clarifier_file)}")

                if os.path.exists(clarifier_file):
                    with open(clarifier_file, 'rb') as f:
                        clarifier_data = pickle.load(f)
                        if DEBUG:
                            print(f"[DEBUG CLARIFIER] {clarifier_file}")
                    return clarifier_data
        except Exception as e:
            print(f"[WARNING] Failed to load clarifier from inferred Thought stage directory: {e}")

    # 策略2: 输出警告并返回空字典
    print(f"[WARNING] No clarifier info found for sample {sample.get('id', 'unknown')}")
    return {}


# ==================== 7. controller_main_loop 函数 ====================

def controller_main_loop(
    sample: Dict[str, Any],
    llm,
    llm_options,
    max_iterations: int = 5,
    cache_dir: Optional[str] = None,
    sample_idx: Optional[int] = None,
    use_clarifier: bool = True,
    thought_results_dir: Optional[str] = None,
    frozen_memory: bool = False,
    use_verifier: bool = False,
    router_variant: Optional[str] = None,
    use_diff_critic: bool = False,
    diff_critic_mode: str = "diagnose_only",
) -> Dict[str, Any]:
    """
    Controller 主循环（兼容现有代码，支持 Clarifier）

    这是阶段一的核心函数，它包装了现有的流程，
    通过 Controller 来决定下一步动作。

    支持默认缓存读取功能（类似 judge_critic_refine_with_cache_mp）：
    - 如果指定了 cache_dir，会尝试从缓存中读取已处理的结果
    - 如果缓存存在则直接返回缓存结果
    - 否则执行完整流程并保存到缓存

    Args:
        sample: 输入样本
        llm: 语言模型实例
        llm_options: 模型选项
        max_iterations: 最大迭代次数
        cache_dir: 缓存目录路径（可选）
        sample_idx: 样本索引（用于缓存文件名）
        use_clarifier: 是否使用 Clarifier 提取模式锚点（默认 True）
        thought_results_dir: Thought 阶段的结果目录路径（用于读取 clarifier 信息）
        frozen_memory: 是否冻结测试期错误树写回

    Returns:
        Dict[str, Any]: 处理后的样本
    """
    
    # 根据 use_clarifier 参数决定是否使用 Clarifier 信息
    # 注意：Refine 阶段从 final_result.pkl 读取的 samples 应该已经包含 clarifier 字段
    # 这个字段在 Thought 阶段由 clarifier.clarify_batch() 添加，并通过 fixed_chain_exec_mp() 保留
    
    if use_clarifier:
        # 使用 Clarifier 信息
        clarifier_info = load_clarifier_info(sample, cache_dir, sample_idx, thought_results_dir)
        sample['clarifier'] = clarifier_info
    else:
        # 不使用 Clarifier 信息
        sample['clarifier'] = {}
    
    # 缓存功能：检查是否存在缓存
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        
        # 确定缓存文件名
        if sample_idx is None:
            sample_idx = sample.get("id", 0) if isinstance(sample.get("id"), int) else 0
        cache_filename = "case-{}.pkl".format(sample_idx)
        cache_path = os.path.join(cache_dir, cache_filename)
        
        # 如果缓存存在，直接加载并返回
        if os.path.exists(cache_path):
            if DEBUG:
                print(f"[CACHE] Loading from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    proc_sample = pickle.load(f)
                return proc_sample
            except Exception as e:
                if DEBUG:
                    print(f"[CACHE] Failed to load cache: {e}, re-executing...")
    
    # 首先调用 Judge 判断初始状态是否正确
    from critic.TableQA.tools import judge_exec_one_sample
    judge_sample = judge_exec_one_sample(
        sample,
        llm=llm,
        llm_options=llm_options
    )
    
    # 初始化状态，使用 Judge 的判断结果
    state = ControllerState(
        sample=judge_sample,
        question=sample.get("statement", ""),
        chain=sample.get("chain", []),
        conclusion=judge_sample.get("judge", "[Incorrect]")
        # conclusion="[Incorrect]"
    )
    
    # 如果初始状态已经正确，直接返回
    if state.is_correct:
        # Record routing info
        judge_sample["_route"] = "SKIP"
        # 缓存正确的结果
        if cache_dir is not None:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(judge_sample, f)
                if DEBUG:
                    print(f"[CACHE] Saved correct result to cache: {cache_path}")
            except Exception as e:
                if DEBUG:
                    print(f"[CACHE] Failed to save cache: {e}")
        return judge_sample

    # AdaRefine Router: decide SKIP/LITE/FULL
    if router_variant:
        from refine.TableQA.utils.router import ROUTER_VARIANTS, RouteDecision
        router_fn = ROUTER_VARIANTS.get(router_variant, ROUTER_VARIANTS["full"])
        route_result = router_fn(judge_sample)

        if DEBUG:
            print(f"[ROUTER] Decision: {route_result}")
            print(f"[ROUTER] Signals: {route_result.signals}")

        if route_result.decision == RouteDecision.SKIP:
            # Already handled above (Judge-Skip)
            pass

        elif route_result.decision == RouteDecision.LITE:
            if DEBUG:
                print(f"[ROUTER] LITE mode: skipping Critic/Controller, direct re-query")
            # LITE: only run simple_query (1 LLM call), skip Critic + Controller
            from refine.TableQA.utils.chain import get_table_info
            from refine.TableQA.operations.final_query import simple_query_cot_original

            table_info = get_table_info(judge_sample, skip_op=[], first_n_op=None)
            lite_sample = simple_query_cot_original(
                judge_sample,
                table_info,
                llm,
                debug=DEBUG,
                use_demo=True,
                llm_options=llm_options
            )

            # Cache and return
            if cache_dir is not None:
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(lite_sample, f)
                    if DEBUG:
                        print(f"[CACHE] Saved LITE result to cache: {cache_path}")
                except Exception as e:
                    if DEBUG:
                        print(f"[CACHE] Failed to save cache: {e}")

            # Record routing info
            lite_sample["_route"] = "LITE"
            lite_sample["_route_signals"] = route_result.signals
            return lite_sample
        else:
            # FULL: continue with standard Controller pipeline
            judge_sample["_route"] = "FULL"
            judge_sample["_route_signals"] = route_result.signals
    else:
        judge_sample["_route"] = "FULL"
        judge_sample["_route_signals"] = {}

    # 初始化组件
    # 修复1：先创建 ActionExecutor 以获取 RetrieverAgent 实例
    executor = ActionExecutor(llm, llm_options, use_verifier=use_verifier, use_diff_critic=use_diff_critic, diff_critic_mode=diff_critic_mode)
    # 将 retriever 传递给 controller，避免重复创建实例
    controller = MinimalController(llm, llm_options, max_iterations, retriever=executor.retriever)
    
    # 主循环说明：
    # 一次迭代执行完整流程：
    # 1. decide() 返回 EXECUTE_TREE
    # 2. 执行 EXECUTE_TREE
    # 3. _llm_decide_diagnose() 决定 DIAGNOSE_BP 或 DIAGNOSE_FS
    # 4. 执行 DIAGNOSE_BP/FS
    # 5. _llm_decide_refine() 决定 REFINE_CHAIN 或 REFINE_QUERY
    # 6. 执行 REFINE_CHAIN/QUERY
    # 7. Judge 检查，如果正确则 UPDATE_TREE 并退出
    
    # 调试模式：打印状态转移
    if DEBUG:
        print("\n" + "="*80)
        print(f"[DEBUG] Starting controller for sample: {sample.get('id', 'unknown')}")
        print("="*80)
    
    for _ in range(max_iterations):
        # 调试打印当前状态
        if DEBUG:
            print(f"\n--- Iteration {state.iteration} ---")
            print(f"Last action: {state.last_action}")
            print(f"Conclusion: {state.conclusion}")
            print(f"__error_route: {state.error_route}")
            print(f"Has diagnosis: {state.diagnosis is not None}")
        
        # 决策
        decision = controller.decide(state)
        
        # 调试打印决策
        if DEBUG:
            print(f"\n>>> Decision: {decision.action.value}")
            print(f"    Reason: {decision.reason}")
            print(f"    Confidence: {decision.confidence}")
            print(f"    Source: {decision.source}")
        
        # 执行
        state = executor.execute(state, decision)
        
        # 检查是否停止
        if decision.action == ControllerAction.STOP:
            if DEBUG:
                print(f"\n!!! STOP: {decision.reason}")
            break
        
        # 使用 LLM 决定下一个动作（根据当前状态动态决策）
        if decision.action == ControllerAction.EXECUTE_TREE:
            # EXECUTE_TREE 后：使用 LLM 决定下一步是 DIAGNOSE_BP 还是 DIAGNOSE_FS
            diagnose_decision = controller._llm_decide_diagnose(state)
            decision = diagnose_decision
            
            # 调试打印
            if DEBUG:
                print(f">>> LLM Diagnose Decision: {decision.action.value}")
                print(f"    Reason: {decision.reason}")
            
            # 执行第二次决策
            state = executor.execute(state, decision)
            
            if decision.action == ControllerAction.STOP:
                break
        
        # DIAGNOSE_BP/DIAGNOSE_FS 后：使用 LLM 决定下一步是 REFINE_CHAIN 还是 REFINE_QUERY
        if decision.action in [ControllerAction.DIAGNOSE_BP, ControllerAction.DIAGNOSE_FS]:
            # 使用 LLM 决定下一步
            refine_decision = controller._llm_decide_refine(state)
            decision = refine_decision
            
            # 调试打印
            if DEBUG:
                print(f">>> LLM Refine Decision: {decision.action.value}")
                print(f"    Reason: {decision.reason}")
            
            # 执行精炼决策
            state = executor.execute(state, decision)
            
            if decision.action == ControllerAction.STOP:
                break
        
        # REFINE_CHAIN/REFINE_QUERY 后：使用 Judge 检查
        if decision.action in [ControllerAction.REFINE_CHAIN, ControllerAction.REFINE_QUERY]:
            from critic.TableQA.tools import judge_exec_one_sample
            judge_sample = judge_exec_one_sample(
                state.sample,
                llm=llm,
                llm_options=llm_options
            )
            state.sample = judge_sample
            state.conclusion = judge_sample.get("judge", "[Incorrect]")
            
            # 调试打印
            if DEBUG:
                print(f">>> Judge Result: {state.conclusion}")
            
            # 如果正确，更新树并停止
            if state.is_correct:
                update_decision = Decision(
                    action=ControllerAction.UPDATE_TREE,
                    reason="Correct answer, update tree",
                    confidence=1.0,
                    source="rule"
                )
                state = executor.execute(state, update_decision)
                if DEBUG:
                    print(f">>> UPDATED TREE!")
                break
    
    # 调试模式：打印最终结果
    if DEBUG:
        print("\n" + "="*80)
        print(f"[DEBUG] Final conclusion: {state.conclusion}")
        print("="*80)
    
    # 缓存功能：保存结果到缓存
    if cache_dir is not None:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(state.sample, f)
            if DEBUG:
                print(f"[CACHE] Saved result to cache: {cache_path}")
        except Exception as e:
            if DEBUG:
                print(f"[CACHE] Failed to save cache: {e}")
    
    return state.sample


# ==================== 7. 便捷函数 ====================

def create_controller_state(sample: Dict[str, Any]) -> ControllerState:
    """
    从样本创建 ControllerState
    
    Args:
        sample: 输入样本
        
    Returns:
        ControllerState: 初始状态
    """
    return ControllerState(
        sample=sample,
        question=sample.get("statement", ""),
        chain=sample.get("chain", []),
        conclusion=sample.get("conclusion", "[Incorrect]") #默认Incorrect
    )


def run_with_controller(
    sample: Dict[str, Any],
    llm,
    llm_options,
    max_iterations: int = 2,
    cache_dir: Optional[str] = None,
    sample_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    使用 Controller 运行样本的便捷函数
    
    Args:
        sample: 输入样本
        llm: 语言模型实例
        llm_options: 模型选项
        max_iterations: 最大迭代次数
        cache_dir: 缓存目录路径（可选）
        sample_idx: 样本索引（用于缓存文件名）
        
    Returns:
        Dict[str, Any]: 处理后的样本
    """
    return controller_main_loop(
        sample=sample,
        llm=llm,
        llm_options=llm_options,
        max_iterations=max_iterations,
        cache_dir=cache_dir,
        sample_idx=sample_idx
    )
