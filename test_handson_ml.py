#!/usr/bin/env python
"""对ageron__handson-ml进行审查测试"""
import sys
import json
from pathlib import Path
from src.core.orchestrator import Orchestrator
from src.core.config import load_config

def test_handson_ml_review():
    try:
        # 加载配置
        cfg = load_config('config.yaml')
        
        # 创建编排器
        orch = Orchestrator(cfg)
        
        # 测试目标仓库
        repo_path = Path('d:\\code_assistant\\Git_repo\\ageron__handson-ml')
        out_dir = Path('d:\\code_assistant\\reports\\ageron__handson-ml')
        
        if not repo_path.exists():
            print(f"❌ 仓库不存在: {repo_path}")
            return False
        
        print(f"🔍 开始审查: ageron__handson-ml")
        print("=" * 60)
        
        # 1. 审查
        print("\n[1/3] 运行审查...")
        review = orch.run_review(repo_path=str(repo_path))
        findings_count = len(review.get('findings', []))
        print(f"  ✅ 审查完成: 发现 {findings_count} 个问题")
        
        # 2. 测试生成
        print("\n[2/3] 生成测试...")
        testgen = orch.run_testgen(repo_path=str(repo_path))
        written_files = testgen.get('written_files', 0)
        print(f"  ✅ 测试生成完成: 生成 {written_files} 个文件")
        
        # 3. 保存结果
        print("\n[3/3] 保存结果...")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        (out_dir / "review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        (out_dir / "testgen.json").write_text(
            json.dumps(testgen, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        print(f"  ✅ 结果已保存: {out_dir}")
        
        # 输出问题统计
        print("\n" + "=" * 60)
        print("📊 问题统计:")
        
        if findings_count > 0:
            # 统计问题类型
            rule_stats = {}
            severity_stats = {}
            
            for finding in review.get('findings', []):
                rule_id = finding.get('rule_id', 'UNKNOWN')
                severity = finding.get('severity', 'UNKNOWN')
                
                rule_stats[rule_id] = rule_stats.get(rule_id, 0) + 1
                severity_stats[severity] = severity_stats.get(severity, 0) + 1
            
            print(f"\n  按严重级别:")
            for severity in ['ERROR', 'WARNING', 'INFO']:
                count = severity_stats.get(severity, 0)
                if count > 0:
                    print(f"    • {severity}: {count}个")
            
            print(f"\n  最常见的问题类型 (Top 10):")
            for rule_id, count in sorted(rule_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    • {rule_id}: {count}个")
            
            print(f"\n  前5个问题详情:")
            for i, f in enumerate(review.get('findings', [])[:5], 1):
                rule_id = f.get('rule_id', 'N/A')
                message = f.get('message', 'N/A')[:50]
                location = f.get('location', {})
                file = Path(location.get('file', 'N/A')).name
                line = location.get('line', 'N/A')
                print(f"    {i}. [{rule_id}] {message}...")
                print(f"       文件: {file}:{line}")
        else:
            print("  ✅ 未发现问题!")
        
        print("\n" + "=" * 60)
        print("✅ ageron__handson-ml 审查测试完成")
        print(f"\n📈 测试指标:")
        print(f"  • 问题总数: {findings_count}")
        print(f"  • 生成测试: {written_files} 个文件")
        print(f"  • 输出目录: {out_dir}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        return False
    except Exception as e:
        print(f"❌ 审查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_handson_ml_review()
    sys.exit(0 if success else 1)
