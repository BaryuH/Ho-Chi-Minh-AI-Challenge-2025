from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")  

collections = utility.list_collections()

for collection in collections:
    utility.drop_collection(collection)
    print(f"Collection {collection} đã bị xóa")
    print("Tất cả các collection đã được xóa.")

