from typing import List, Dict, Any, Optional,Union



def print_list_type(list_type: List[int]) -> None:
    """
    打印列表类型
    """
    print(list_type)
    for i in list_type:
        print(i*2)

print_list_type([1, 2, 3])

def print_dict_type(dict_type: Dict[str, int]) -> None:
    """
    打印字典类型
    """
    print(dict_type)
    for key, value in dict_type.items():
        print(key, value)

print_dict_type({"a": 1, "b": 2})

def print_any_type(any_type: Any) -> None:
    """
    打印任意类型
    """
    print(any_type)     

print_any_type("hello")
print_any_type(123)
print_any_type([1, 2, 3])
print_any_type({"a": 1, "b": 2})


def print_optional_type(optional_type: Optional[str]) -> str:
    """
    打印可选类型
    """
    if optional_type == "name":
        print("name")
        return "name"

    print(optional_type)
    return None


print_optional_type("name")
#

def print_union_type(union_type: Union[int, str]) -> str:
    """
    打印联合类型    
    """
    print(union_type)
    return union_type

print_union_type(12)
print_union_type("hello")
