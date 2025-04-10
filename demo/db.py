from operator import is_
import mysql.connector


class MySQLDatabase:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

        def connect(self):
            try:  # 连接到MySQL数据库
                self.connection = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database
                )
                if self.connection.is_connected():
                    print("成功 Connected to MySQL database")
            except mysql.connector.Error as e:
                print(f"失败 connecting to MySQL database: {e}")


        def execute_query(self,query,params=None):
            try:
                cursor = self.connection.cursor()
                cursor.execute(query,params)
                self.connection.commit()
                return cursor
            except mysql.connector.Error as e:
                print(f"Error executing query: {e}")
                return None


        def fetch_all(self,cursor):
            try:
                return cursor.fetchall()
            except mysql.connector.Error as e:
                print(f"Error fetching all results: {e}")
                return None
        def fetch_one(self,cursor):
            try:
                return cursor.fetchone()
            except mysql.connector.Error as e:
                print(f"Error fetching one results: {e}")
                return None
        def close(self):
            try:
                self.connection.close()
                print("MySQL connection closed")
            except mysql.connector.Error as e:
                print(f"Error closing MySQL connection: {e}")       

if __name__ == "__main__":
    print("hello")
    db = MySQLDatabase("192.168.0.98", "root", "rootbeyond123", "ivics")
    db.connect()
    cursor = db.execute_query("SELECT * FROM users")
    results = db.fetch_all(cursor)
    print(results)
    db.close()
    print("world")

    