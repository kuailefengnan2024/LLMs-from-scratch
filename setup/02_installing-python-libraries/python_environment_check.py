# 版权所有 (c) Sebastian Raschka，遵循Apache License 2.0 (详见LICENSE.txt)。
# 来源于 "从零开始构建大语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

from importlib.metadata import PackageNotFoundError, import_module, version as get_version
from os.path import dirname, exists, join, realpath
from packaging.version import parse as version_parse
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import platform
import sys

if version_parse(platform.python_version()) < version_parse("3.9"):
    print("[失败] 我们推荐Python 3.9或更新版本，但发现版本为 %s" % sys.version)
else:
    print("[成功] 您的Python版本是 %s" % platform.python_version())


def get_packages(pkgs):
    """
    返回一个字典，将包名（小写）映射到其已安装的版本。
    """
    PACKAGE_MODULE_OVERRIDES = {
        "tensorflow-cpu": ["tensorflow", "tensorflow_cpu"],
    }
    result = {}
    for p in pkgs:
        # 确定要尝试的可能模块名。
        module_names = PACKAGE_MODULE_OVERRIDES.get(p.lower(), [p])
        version_found = None
        for module_name in module_names:
            try:
                imported = import_module(module_name)
                version_found = getattr(imported, "__version__", None)
                if version_found is None:
                    try:
                        version_found = get_version(module_name)
                    except PackageNotFoundError:
                        version_found = None
                if version_found is not None:
                    break  # 如果成功获取版本就停止。
            except ImportError:
                # 还尝试将连字符替换为下划线作为备用方案。
                alt_module = module_name.replace("-", "_")
                if alt_module != module_name:
                    try:
                        imported = import_module(alt_module)
                        version_found = getattr(imported, "__version__", None)
                        if version_found is None:
                            try:
                                version_found = get_version(alt_module)
                            except PackageNotFoundError:
                                version_found = None
                        if version_found is not None:
                            break
                    except ImportError:
                        continue
                continue
        if version_found is None:
            version_found = "0.0"
        result[p.lower()] = version_found
    return result


def get_requirements_dict():
    """
    解析requirements.txt并返回一个字典，将包名（小写）
    映射到指定符字符串（例如">=2.18.0,<3.0"）。它使用packaging.requirements中的Requirement类
    来正确处理环境标记，并将每个对象的指定符转换为字符串。
    """

    PROJECT_ROOT = dirname(realpath(__file__))
    PROJECT_ROOT_UP_TWO = dirname(dirname(PROJECT_ROOT))
    REQUIREMENTS_FILE = join(PROJECT_ROOT_UP_TWO, "requirements.txt")
    if not exists(REQUIREMENTS_FILE):
        REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

    reqs = {}
    with open(REQUIREMENTS_FILE) as f:
        for line in f:
            # 移除内联注释和尾随空白字符。
            # 这在第一个'#'处分割并取之前的部分。
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                req = Requirement(line)
            except Exception as e:
                print(f"由于解析错误跳过行: {line} ({e})")
                continue
            # 如果存在标记则评估标记。
            if req.marker is not None and not req.marker.evaluate():
                continue
            # 存储包名及其版本指定符。
            spec = str(req.specifier) if req.specifier else ">=0"
            reqs[req.name.lower()] = spec
    return reqs


def check_packages(reqs):
    """
    检查已安装的包版本是否符合要求。
    """
    installed = get_packages(reqs.keys())
    for pkg_name, spec_str in reqs.items():
        spec_set = SpecifierSet(spec_str)
        actual_ver = installed.get(pkg_name, "0.0")
        if actual_ver == "N/A":
            continue
        actual_ver_parsed = version_parse(actual_ver)
        # 如果已安装版本是预发布版本，则在指定符中允许预发布版本。
        if actual_ver_parsed.is_prerelease:
            spec_set.prereleases = True
        if actual_ver_parsed not in spec_set:
            print(f"[失败] {pkg_name} {actual_ver_parsed}，请安装匹配 {spec_set} 的版本")
        else:
            print(f"[成功] {pkg_name} {actual_ver_parsed}")


def main():
    reqs = get_requirements_dict()
    check_packages(reqs)


if __name__ == "__main__":
    main()
