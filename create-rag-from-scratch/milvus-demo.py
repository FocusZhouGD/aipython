from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker

# ==================== 连接服务 ====================
client = MilvusClient(uri="http://localhost:19530")  # 连接本地服务

# ==================== 创建集合 ====================
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
client.create_collection(collection_name="demo_v4", schema=schema)

# ==================== 创建索引 ====================
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="",  # 根据需求填写具体索引类型
    index_name="vector_index"
)
client.create_index(collection_name="demo_v4", index_params=index_params)

# ==================== 数据操作 ====================
# 插入初始数据
insert_data = [
    {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"}
]
client.insert(collection_name="demo_v4", data=insert_data)

# 更新/插入数据
upsert_data = [
    {"id": 0, "vector": [-0.619954382375778, 0.4479436794798608, -0.17493894838751745, -0.4248030059917294, -0.8648452746018911], "color": "black_9898"},
    {"id": 1, "vector": [0.4762662251462588, -0.6942502138717026, -0.4490002642657902, -0.628696575798281, 0.9660395877041965], "color": "red_7319"}
]
client.upsert(collection_name='demo_v4', data=upsert_data)

# 删除数据
client.delete(collection_name="demo_v4", filter="id in [4,5,6]")  # 过滤删除
client.delete(collection_name="demo_v4", ids=[18, 19])           # ID删除

# ==================== 搜索操作 ====================
# 单向量搜索
search_res = client.search(
    collection_name="demo_v4",
    data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
    limit=2,
    search_params={"metric_type": "IP", "params": {}}
)

# 带过滤条件搜索
filter_res = client.search(
    collection_name="demo_v4",
    data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
    limit=5,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["color"],
    filter='color like "gree%"'
)

# 多向量混合搜索（高级用法）
# 定义两个搜索请求
query_vector1 = [[0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]
search_param1 = {
    "data": query_vector1,
    "anns_field": "vector",
    "param": {"metric_type": "L2", "params": {"nprobe": 10}},
    "limit": 2
}

query_vector2 = [[0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]
search_param2 = {
    "data": query_vector2,
    "anns_field": "vector",
    "param": {"metric_type": "L2", "params": {"nprobe": 10}},
    "limit": 2
}

# 执行混合搜索
hybrid_res = client.hybrid_search(
    requests=[AnnSearchRequest(&zwnj;**search_param1), AnnSearchRequest(**&zwnj;search_param2)],
    rerank=WeightedRanker(0.8, 0.2),  # 设置权重
    limit=2
)

print("搜索完成！")
