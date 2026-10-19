"""
Constraint Induction Module
将Critic的判别结果转化为结构化约束
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from .constraint_state import ConstraintState


class ConstraintInducer:
    """
    约束归纳器，将Critic的判别结果转化为结构化约束
    
    核心功能：
    1. 从自然语言critique中提取错误类型
    2. 识别结构定位（行、列、操作）
    3. 生成相应的约束规则
    4. 更新约束状态
    """
    
    def __init__(self, llm=None):
        """
        初始化约束归纳器
        
        Args:
            llm: LLM实例，用于复杂情况下的结构化提取
        """
        self.llm = llm
        
        # 错误类型的正则表达式模式
        self.error_type_patterns = {
            "row_error": [
                r"row (\d+) was omitted",
                r"misses? (?:the )?row (\d+)",
                r"incorrectly includes? row (\d+)",
                r"fails? to include row (\d+)",
                r"overlooks? row (\d+)",
                r"wrongly includes? row (\d+)",
                r"should not include row (\d+)"
            ],
            "column_error": [
                r"incorrectly filters? out(?:\s+the\s+)?column[s]?\s+(.+?)(?:\.|,)",
                r"omits? (?:the\s+)?column[s]?\s+(.+?)(?:\.|,)",
                r"should (?:not\s+)?filter out(?:\s+the\s+)?column[s]?\s+(.+?)(?:\.|,)",
                r"fails? to include(?:\s+the\s+)?column[s]?\s+(.+?)(?:\.|,)",
                r"wrongly filters? out(?:\s+the\s+)?column[s]?\s+(.+?)(?:\.|,)"
            ],
            "aggregation_error": [
                r"incorrectly groups? by\s+(.+?)(?:\.|,)",
                r"should group by\s+(.+?)(?:\.|,)",
                r"wrong aggregation scope\s+(.+?)(?:\.|,)",
                r"incorrect\s+grouping\s+by\s+(.+?)(?:\.|,)"
            ],
            "entity_confusion": [
                r"confuses?\s+(.+?)\s+with\s+(.+?)(?:\.|,)",
                r"misidentif(?:y|ies)?\s+(.+?)(?:\.|,)",
                r"mixes?\s+up\s+(.+?)\s+and\s+(.+?)(?:\.|,)"
            ]
        }
    
    def induce_constraints(
        self, 
        critique: str, 
        incorrect_step: int, 
        sample: Dict,
        constraint_state: ConstraintState,
        debug: bool = False
    ) -> Tuple[ConstraintState, Dict[str, Any]]:
        """
        从critique中诱导约束并更新约束状态
        
        Args:
            critique: Critic的自然语言判别结果
            incorrect_step: 错误步骤编号
            sample: 样本数据
            constraint_state: 当前约束状态
            debug: 是否启用调试
            
        Returns:
            Tuple[ConstraintState, Dict]: 更新后的约束状态和诱导结果
        """
        # 1. 分析critique，提取错误类型和结构定位
        error_analysis = self._analyze_critique(critique, incorrect_step, sample)
        
        # 2. 根据错误类型生成相应的约束
        induced_constraints = []
        
        if error_analysis["error_type"] == "row_error":
            induced_constraints = self._induce_row_constraints(
                error_analysis, constraint_state, debug
            )
        elif error_analysis["error_type"] == "column_error":
            induced_constraints = self._induce_column_constraints(
                error_analysis, constraint_state, debug
            )
        elif error_analysis["error_type"] == "aggregation_error":
            induced_constraints = self._induce_aggregation_constraints(
                error_analysis, constraint_state, debug
            )
        elif error_analysis["error_type"] == "entity_confusion":
            induced_constraints = self._induce_entity_constraints(
                error_analysis, constraint_state, debug
            )
        
        # 3. 增加推理轮次
        constraint_state.increment_round()
        
        # 4. 返回更新后的状态和诱导结果
        induction_result = {
            "error_analysis": error_analysis,
            "induced_constraints": induced_constraints,
            "constraint_summary": constraint_state.get_constraint_summary()
        }
        
        return constraint_state, induction_result
    
    def _analyze_critique(
        self, 
        critique: str, 
        incorrect_step: int, 
        sample: Dict
    ) -> Dict[str, Any]:
        """
        分析critique，提取错误类型和结构定位
        
        Args:
            critique: Critic的判别结果
            incorrect_step: 错误步骤编号
            sample: 样本数据
            
        Returns:
            Dict: 包含错误分析信息的字典
        """
        analysis = {
            "error_type": None,
            "structural_localization": {},
            "source_step": incorrect_step,
            "critique_snippet": critique[:200] + "..." if len(critique) > 200 else critique
        }
        
        # 获取错误步骤的操作信息
        if incorrect_step > 0 and incorrect_step <= len(sample.get("chain", [])):
            error_operation = sample["chain"][incorrect_step - 1]
            analysis["error_operation"] = error_operation
            analysis["operation_name"] = error_operation["operation_name"]
            analysis["operation_parameters"] = error_operation.get("parameter_and_conf", [])
        else:
            analysis["error_operation"] = None
            analysis["operation_name"] = None
            analysis["operation_parameters"] = []
        
        # 使用正则表达式匹配错误类型
        for error_type, patterns in self.error_type_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, critique, re.IGNORECASE)
                if matches:
                    analysis["error_type"] = error_type
                    analysis["structural_localization"]["matches"] = matches
                    analysis["structural_localization"]["matched_pattern"] = pattern
                    break
            if analysis["error_type"]:
                break
        
        # 如果正则匹配失败，使用LLM进行结构化提取
        if not analysis["error_type"] and self.llm:
            analysis = self._llm_based_analysis(critique, incorrect_step, sample)
        
        return analysis
    
    def _llm_based_analysis(
        self, 
        critique: str, 
        incorrect_step: int, 
        sample: Dict
    ) -> Dict[str, Any]:
        """
        使用LLM进行结构化分析
        
        Args:
            critique: Critic的判别结果
            incorrect_step: 错误步骤编号
            sample: 样本数据
            
        Returns:
            Dict: 包含错误分析信息的字典
        """
        # 构建错误操作的上下文
        operation_context = ""
        if incorrect_step > 0 and incorrect_step <= len(sample.get("chain", [])):
            error_operation = sample["chain"][incorrect_step - 1]
            operation_context = f"""
Error Operation:
- Operation Name: {error_operation['operation_name']}
- Operation Parameters: {error_operation.get('parameter_and_conf', [])}
- Thought: {error_operation.get('thought', 'N/A')}
"""
        
        prompt = f"""You are a constraint extraction expert. Analyze the following critique and extract structured information about the error.

Critique:
{critique}

{operation_context}

Extract the following information in JSON format:
{{
    "error_type": "row_error|column_error|aggregation_error|entity_confusion",
    "structural_localization": {{
        "forbidden_rows": ["row 1", "row 2", ...],
        "forbidden_columns": ["column_name", ...],
        "suggested_correction": "...",
        "reasoning": "..."
    }},
    "confidence": 0.0-1.0
}}

Output ONLY valid JSON, no additional text."""

        try:
            # 使用正确的LLM方法
            responses = self.llm.generate_plus_with_score(
                prompt, 
                options=self.llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=500,
                    per_example_top_p=1.0
                )
            )
            response = responses[0][0] if responses else ""
            
            # 清理响应，提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                analysis["source_step"] = incorrect_step
                analysis["critique_snippet"] = critique[:200] + "..." if len(critique) > 200 else critique
                
                # 添加错误操作信息
                if incorrect_step > 0 and incorrect_step <= len(sample.get("chain", [])):
                    error_operation = sample["chain"][incorrect_step - 1]
                    analysis["error_operation"] = error_operation
                    analysis["operation_name"] = error_operation["operation_name"]
                    analysis["operation_parameters"] = error_operation.get("parameter_and_conf", [])
                
                return analysis
        except Exception as e:
            print(f"LLM-based analysis failed: {e}")
        
        # 如果LLM分析失败，返回默认分析
        return {
            "error_type": None,
            "structural_localization": {},
            "source_step": incorrect_step,
            "critique_snippet": critique[:200] + "..." if len(critique) > 200 else critique
        }
    
    def _induce_row_constraints(
        self, 
        error_analysis: Dict, 
        constraint_state: ConstraintState
    ) -> List[Dict[str, Any]]:
        """
        诱导行约束
        
        Args:
            error_analysis: 错误分析结果
            constraint_state: 约束状态
            
        Returns:
            List[Dict]: 诱导的约束列表
        """
        induced_constraints = []
        matches = error_analysis["structural_localization"].get("matches", [])
        
        # 提取被错误选择的行
        forbidden_rows = []
        
        # 从正则匹配中提取
        for match in matches:
            if isinstance(match, str):
                # 处理 "row 8" 格式
                row_match = re.search(r"row (\d+)", match, re.IGNORECASE)
                if row_match:
                    row_num = row_match.group(1)
                    forbidden_rows.append(f"row {row_num}")
            elif isinstance(match, list):
                for m in match:
                    row_match = re.search(r"row (\d+)", m, re.IGNORECASE)
                    if row_match:
                        row_num = row_match.group(1)
                        forbidden_rows.append(f"row {row_num}")
        
        # 检查LLM分析结果
        if "forbidden_rows" in error_analysis["structural_localization"]:
            llm_rows = error_analysis["structural_localization"]["forbidden_rows"]
            forbidden_rows.extend(llm_rows)
        
        # 去重
        forbidden_rows = list(set(forbidden_rows))
        
        if forbidden_rows:
            constraint_state.add_forbidden_rows(
                forbidden_rows,
                error_analysis["error_type"],
                error_analysis["source_step"]
            )
            induced_constraints.append({
                "type": "forbidden_rows",
                "value": forbidden_rows,
                "reason": "Row selection error detected"
            })
        
        return induced_constraints
    
    def _induce_column_constraints(
        self, 
        error_analysis: Dict, 
        constraint_state: ConstraintState
    ) -> List[Dict[str, Any]]:
        """
        诱导列约束
        
        Args:
            error_analysis: 错误分析结果
            constraint_state: 约束状态
            
        Returns:
            List[Dict]: 诱导的约束列表
        """
        induced_constraints = []
        matches = error_analysis["structural_localization"].get("matches", [])
        
        # 提取被错误过滤的列
        forbidden_columns = []
        
        # 从正则匹配中提取
        for match in matches:
            if isinstance(match, str):
                # 清理列名
                column_name = match.strip().strip('"\'').strip()
                if column_name:
                    forbidden_columns.append(column_name)
        
        # 检查LLM分析结果
        if "forbidden_columns" in error_analysis["structural_localization"]:
            llm_cols = error_analysis["structural_localization"]["forbidden_columns"]
            forbidden_columns.extend(llm_cols)
        
        # 去重
        forbidden_columns = list(set(forbidden_columns))
        
        if forbidden_columns:
            constraint_state.add_forbidden_columns(
                forbidden_columns,
                error_analysis["error_type"],
                error_analysis["source_step"]
            )
            induced_constraints.append({
                "type": "forbidden_columns",
                "value": forbidden_columns,
                "reason": "Column selection error detected"
            })
        
        return induced_constraints
    
    def _induce_aggregation_constraints(
        self, 
        error_analysis: Dict, 
        constraint_state: ConstraintState
    ) -> List[Dict[str, Any]]:
        """
        诱导聚合约束
        
        Args:
            error_analysis: 错误分析结果
            constraint_state: 约束状态
            
        Returns:
            List[Dict]: 诱导的约束列表
        """
        induced_constraints = []
        matches = error_analysis["structural_localization"].get("matches", [])
        
        # 提取正确的聚合列
        correct_columns = []
        
        # 从正则匹配中提取
        for match in matches:
            if isinstance(match, str):
                column_name = match.strip().strip('"\'').strip()
                if column_name:
                    correct_columns.append(column_name)
        
        # 检查LLM分析结果
        if "suggested_correction" in error_analysis["structural_localization"]:
            correction = error_analysis["structural_localization"]["suggested_correction"]
            if correction:
                correct_columns.append(correction)
        
        # 去重
        correct_columns = list(set(correct_columns))
        
        if correct_columns:
            # 使用第一个建议的修正
            constraint_state.add_aggregation_constraint(
                "group_column",
                correct_columns[0],
                error_analysis["error_type"],
                error_analysis["source_step"]
            )
            induced_constraints.append({
                "type": "aggregation",
                "value": {"group_column": correct_columns[0]},
                "reason": "Aggregation error detected"
            })
        
        return induced_constraints
    
    def _induce_entity_constraints(
        self, 
        error_analysis: Dict, 
        constraint_state: ConstraintState,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        诱导实体约束
        
        Args:
            error_analysis: 错误分析结果
            constraint_state: 约束状态
            debug: 是否启用调试
            
        Returns:
            List[Dict]: 诱导的约束列表
        """
        induced_constraints = []
        
        # 实体混淆通常需要禁止特定的行选择模式
        if "error_operation" in error_analysis:
            operation_name = error_analysis["operation_name"]
            operation_parameters = error_analysis["operation_parameters"]
            
            if operation_name == "select_row" and operation_parameters:
                # 提取被错误选择的行
                selected_rows = []
                for param, conf in operation_parameters[:2]:  # 取前两个最高置信度的
                    try:
                        # 改进参数解析逻辑
                        if isinstance(param, list):
                            # param是列表格式：[["row 1", "row 2"], 0.9]
                            param_value = param[0]
                        elif isinstance(param, str):
                            # param是字符串格式：'["row 1", "row 2"]'
                            param_value = param
                        else:
                            continue
                        
                        # 解析行号
                        if isinstance(param_value, str):
                            # 处理字符串形式的参数
                            import ast
                            try:
                                parsed_rows = ast.literal_eval(param_value)
                                if isinstance(parsed_rows, list):
                                    selected_rows.extend(parsed_rows)
                                elif isinstance(parsed_rows, (int, str)):
                                    selected_rows.append(parsed_rows)
                            except:
                                pass
                        elif isinstance(param_value, (list, tuple)):
                            # 处理列表或元组形式的参数
                            for item in param_value:
                                if isinstance(item, int):
                                    selected_rows.append(item)
                                elif isinstance(item, str) and item.strip().isdigit():
                                    selected_rows.append(int(item.strip()))
                    except Exception as e:
                        if debug:
                            print(f"Warning: Failed to parse parameter {param}: {e}")
                
                # 去重
                selected_rows = list(set(selected_rows))
                
                if selected_rows:
                    row_names = [f"row {r}" for r in selected_rows]
                    constraint_state.add_forbidden_operation(
                        operation_name,
                        row_names,
                        error_analysis["error_type"],
                        error_analysis["source_step"]
                    )
                    induced_constraints.append({
                        "type": "forbidden_operation",
                        "value": (operation_name, row_names),
                        "reason": "Entity confusion detected"
                    })
        
        return induced_constraints
    
    def get_error_type_statistics(self) -> Dict[str, int]:
        """
        获取错误类型统计（需要外部维护）
        
        Returns:
            Dict: 错误类型统计
        """
        # 这个方法需要在实际使用中维护统计信息
        return {
            "row_error": 0,
            "column_error": 0,
            "aggregation_error": 0,
            "entity_confusion": 0
        }
