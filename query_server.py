import sys
import numpy as np
import grpc

sys.path.append("python")
import faiss_pb2, faiss_pb2_grpc

ENCODINGS_PATH = "triviaqa_encodings.npy"
TOP_K = 5
SERVER_ADDR = "localhost:8080"

def main():
    vectors = np.load(ENCODINGS_PATH)
    vectors = np.asarray(vectors, dtype=np.float32)

    # Use first 10 vectors
    vectors = vectors[:10]

    channel = grpc.insecure_channel(SERVER_ADDR)
    stub = faiss_pb2_grpc.FaissServiceStub(channel)

    for i, v in enumerate(vectors):
        req = faiss_pb2.SearchRequest(
            vector=faiss_pb2.Vector(float_val=v.tolist()),
            top_k=TOP_K,
        )
        resp = stub.Search(req)
        print(f"Query {i}: {resp}")

if __name__ == "__main__":
    main()