

from typing import Optional


def get_user_by_email(email) ->Optional[dict]:
 try:
    # 尝试从数据库中获取用户信息
    with get_db() as db:

        user = db.query("SELECT * FROM users WHERE email = %s", (email,)).first()
    

    return user
 except Exception as e:
    # 如果发生异常，打印错误信息
    print(f"Error fetching user: {e}")
    return None