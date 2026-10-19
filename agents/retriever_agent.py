# Copyright 2024 Table-Critic contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Retriever Agent: 独立的检索器 Agent

该模块实现了从 CriticAgent 中提取的检索功能，用于：
1. 根据 error_route 检索 blueprint 和 few-shot 示例
2. 支持基于相似度的检索（未来扩展）

这是阶段2的核心组件，将检索逻辑从 CriticAgent 中解耦出来。

该实现复用了 critic/TableFV/tools/get_info.py 中的检索逻辑：
- get_critic_few_shot: 检索 few-shot 示例
- return_error_shot: 根据 error_route 返回相应示例
- get_terminal_nodes: 处理 blueprint 的提取
"""

import json
import random
import copy
import traceback
from typing import Dict, List, Any, Optional
from .multi_agent_framework import BaseAgent, AgentType


# 设置随机种子以保证可复现性
random.seed(42)


def get_terminal_nodes(input_dict: Dict, selected_blueprint: bool = False) -> List[str]:
    """
    从错误树中提取终端节点
    
    Args:
        input_dict: 错误树字典
        selected_blueprint: 是否只选择 blueprint
        
    Returns:
        终端节点列表
    """
    terminal_nodes = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            terminal_nodes.extend(get_terminal_nodes(value, selected_blueprint))
        elif isinstance(value, list):
            # 处理列表项 - 可以是字符串或字典
            for item in value:
                if isinstance(item, dict):
                    # 新格式: 带有 'content' 字段的字典
                    if selected_blueprint and 'blueprint' in item:
                        terminal_nodes.append(item['blueprint'])
                    elif 'content' in item:
                        terminal_nodes.append(item['content'])
                    else:
                        # 回退: 将字典转换为字符串
                        terminal_nodes.append(str(item))
                else:
                    # 旧格式: 字符串
                    terminal_nodes.append(item)
        else:
            # 直接字符串值
            terminal_nodes.append(value)
    return terminal_nodes


def return_error_shot(
    error_route: str,
    few_shot_dict: Dict,
    selected_blueprint: bool = False
) -> List[Any]:
    """
    根据 error_route 返回相应的 few-shot 示例
    
    Args:
        error_route: 错误分类路径
        few_shot_dict: few-shot 字典
        selected_blueprint: 是否只选择 blueprint
        
    Returns:
        few-shot 示例列表
    """
    if error_route != 'random':
        error_route_parts = error_route.split('->')
        few_shot = copy.deepcopy(few_shot_dict)
        for error_type in error_route_parts:
            error_type = error_type.strip()
            if error_type in few_shot:
                few_shot = few_shot[error_type]
                if isinstance(few_shot, list):
                    if len(few_shot) > 5:
                        few_shot = random.sample(few_shot, 5)
                        new_few_shot = []
                        for item in few_shot:
                            if isinstance(item, dict):
                                if selected_blueprint and 'blueprint' in item:
                                    new_few_shot.append(item['blueprint'])
                                elif 'content' in item:
                                    new_few_shot.append(item['content'])
                                else:
                                    new_few_shot.append(str(item))
                            else:
                                new_few_shot.append(item)
                        few_shot = new_few_shot
                    return few_shot
    
    # 如果没有正确返回，则随机选择
    few_shot = get_terminal_nodes(few_shot_dict, selected_blueprint)
    few_shot = random.sample(few_shot, min(5, len(few_shot)))
    return few_shot


def get_critic_few_shot(
    error_route: str,
    few_shot_json: str = None,
    selected_blueprint: bool = False
) -> str:
    """
    获取 critic few-shot 示例（与 get_info.py 中的实现一致）
    
    Args:
        error_route: 错误分类路径
        few_shot_json: few-shot JSON 文件路径
        selected_blueprint: 是否只选择 blueprint
        
    Returns:
        格式化的 few-shot 字符串
    """
    few_shot = "\nHere are some examples.\n\n"
    
    try:
        with open(few_shot_json, 'r') as f:
            few_shot_dict = json.load(f)
            selected_few_shot = return_error_shot(error_route, few_shot_dict, selected_blueprint)
        
        random.shuffle(selected_few_shot)
        
        for idx, shot in enumerate(selected_few_shot):
            if isinstance(shot, dict):
                if selected_blueprint and 'blueprint' in shot:
                    few_shot += f"Example {idx+1}:\n" + shot['blueprint'] + "\n\n\n"
                elif 'content' in shot:
                    few_shot += f"Example {idx+1}:\n" + shot['content'] + "\n\n\n"
                else:
                    few_shot += f"Example {idx+1}:\n" + str(shot) + "\n\n\n"
            else:
                few_shot += f"Example {idx+1}:\n" + str(shot) + "\n\n\n"
    except Exception as e:
        print(f"Warning: Could not load few-shot from {few_shot_json}: {e}")
        return ""
    
    return few_shot


class RetrieverAgent(BaseAgent):
    """
    独立的检索器 Agent
    
    负责从错误树（error tree）中检索相关的 blueprint 和 few-shot 示例。
    该 Agent 将检索功能从 CriticAgent 中解耦，支持独立调用。
    
    主要功能：
    - retrieve_by_route: 根据 error_route 检索 blueprint 和 few-shot
    - retrieve_blueprint_only: 仅检索 blueprint
    - retrieve_few_shot_only: 仅检索 few-shot 示例
    - retrieve_by_similarity: 基于相似度检索（未来扩展）
    """

    def __init__(
        self,
        llm=None,
        memory_path: str = None
    ):
        """
        初始化 RetrieverAgent
        
        Args:
            llm: 语言模型实例（可选，用于未来扩展的语义检索）
            memory_path: 错误树/蓝图 memory 文件路径
        """
        super().__init__(AgentType.RETRIEVER, llm)
        self.memory_path = memory_path
        # 使用单一缓存字典存储完整结果，避免不一致
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def process(self, sample: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理样本并返回检索结果（兼容 BaseAgent 接口）
        
        Args:
            sample: 输入样本
            context: 上下文，应包含 error_route
            
        Returns:
            包含检索结果的样本
        """
        error_route = 'random'
        if context is not None:
            error_route = context.get('error_route', 'random')
        
        # 执行检索
        retrieval_result = self.retrieve_by_route(error_route)
        
        # 将检索结果添加到样本中
        sample['retrieved_blueprint'] = retrieval_result.get('blueprint', '')
        sample['retrieved_few_shot'] = retrieval_result.get('few_shot_examples', [])
        sample['retrieved_route'] = error_route
        
        return sample
    
    def retrieve_by_route(self, error_route: str) -> Dict[str, Any]:
        """
        根据 error_route 检索 blueprint 和 few-shot 示例
        
        这是核心检索方法，根据错误分类路径从错误树中检索相关信息。
        
        Args:
            error_route: 错误分类路径，如 "sub-table error" 或 "final query error"
            
        Returns:
            包含 blueprint 和 content 的字典
        """
        # 检查缓存
        if error_route in self.cache:
            return self.cache[error_route]
        
        result = {
            'blueprint': None,
            'few_shot_examples': []
        }
        
        try:
            # 加载错误树
            with open(self.memory_path, 'r') as f:
                error_tree = json.load(f)
            
            # 获取 few-shot 示例（包含 blueprint）
            few_shot_list = return_error_shot(error_route, error_tree, selected_blueprint=False)
            
            # 获取 blueprint only
            blueprint_list = return_error_shot(error_route, error_tree, selected_blueprint=True)
            
            if blueprint_list:
                result['blueprint'] = blueprint_list[0] if blueprint_list else None
            
            result['few_shot_examples'] = few_shot_list
            
            # 只缓存有效的检索结果（非空列表），避免空列表被缓存导致后续检索失败
            if few_shot_list or blueprint_list:
                self.cache[error_route] = result
            
        except Exception as e:
            print(f"RetrieverAgent error: {e}", flush=True)
            print(f"Full traceback: {traceback.format_exc()}", flush=True)
        
        return result
    
    def retrieve_blueprint_only(self, error_route: str) -> str:
        """
        仅检索 blueprint（不包含 few-shot 示例）
        
        Args:
            error_route: 错误分类路径
            
        Returns:
            Blueprint 文本
        """
        # 检查缓存
        if error_route in self.cache:
            cached = self.cache[error_route].get('blueprint')
            return cached if cached is not None else ''
        
        result = self.retrieve_by_route(error_route)
        blueprint = result.get('blueprint')
        return blueprint if blueprint is not None else ''
    
    def retrieve_few_shot_only(
        self,
        error_route: str,
        num_examples: int = 3,
        include_blueprint: bool = True
    ) -> str:
        """
        检索 few-shot 示例（可选择是否包含 blueprint）
        
        这个方法与 get_critic_few_shot 函数功能一致
        
        Args:
            error_route: 错误分类路径
            num_examples: 要检索的示例数量
            include_blueprint: 是否在示例中包含 blueprint
            
        Returns:
            格式化的 few-shot prompt 字符串
        """
        return get_critic_few_shot(
            error_route,
            few_shot_json=self.memory_path,
            selected_blueprint=include_blueprint
        )
    
    def retrieve_by_similarity(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        基于相似度检索（未来扩展）
        
        TODO: 实现向量检索功能
        - 使用 embedding 模型计算query与示例的相似度
        - 返回 top_k 个最相似的示例
        
        Args:
            query: 查询文本
            top_k: 返回的相似示例数量
            
        Returns:
            相似示例列表（当前版本返回空列表）
        """
        # TODO: 实现向量检索
        # 1. 如果没有 llm，返回空列表
        # 2. 使用 embedding 模型计算query与示例的相似度
        # 3. 返回 top_k 个最相似的示例
        
        return []
    
    def build_retrieval_prompt(
        self,
        sample: Dict[str, Any],
        error_route: str,
        use_few_shot: bool = True
    ) -> str:
        """
        构建包含检索结果的 prompt（供 CriticAgent 使用）
        
        这个方法与 CriticAgent._build_critique_prompt 中的检索部分功能一致
        
        Args:
            sample: 当前样本
            error_route: 错误分类路径
            use_few_shot: 是否包含 few-shot 示例
            
        Returns:
            完整的 prompt 字符串
        """
        # 检索 blueprint
        blueprint = self.retrieve_blueprint_only(error_route)
        
        prompt = ""
        
        if blueprint:
            prompt += f"\n--- Error Pattern Blueprint ---\n"
            prompt += f"Error Route: {error_route}\n"
            prompt += f"Pattern Summary: {blueprint}\n"
            prompt += "--- End Blueprint ---\n\n"
        
        # 检索 few-shot 示例
        if use_few_shot:
            few_shot_examples = self.retrieve_few_shot_only(
                error_route,
                num_examples=3,
                include_blueprint=False
            )
            if few_shot_examples:
                prompt += few_shot_examples
        
        return prompt
    
    def clear_cache(self) -> None:
        """
        清除缓存
        """
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        return {
            'cache_size': len(self.cache)
        }
