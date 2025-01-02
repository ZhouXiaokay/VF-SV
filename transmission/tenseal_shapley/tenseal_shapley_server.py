import time
from concurrent import futures
import sys
code_path = '/home/zxk/codes/vfps_mi_diversity'
sys.path.append(code_path)
import numpy as np
import grpc
import tenseal as ts
from conf.args import global_args_parser

from utils.helpers import get_utility_key, utility_key_to_groups

import tenseal_allreduce_data_pb2_grpc, tenseal_allreduce_data_pb2
import tenseal_shapley_data_pb2_grpc, tenseal_shapley_data_pb2


class ShapleyServer(tenseal_shapley_data_pb2_grpc.MIServiceServicer):
    def __init__(self, address, num_clients, ctx_file):
        self.address = address
        self.num_clients = num_clients

        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.sleep_time = 0.001

        # cache and counter for sum operation
        self.n_sum_round = 0
        self.sum_enc_vectors_dict = {}
        self.sum_enc_vectors_list = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

        print("Shapley server has been initialized")
        print("world size is:", num_clients )

    def reset_sum(self):
        self.sum_enc_vectors_dict = {}
        self.sum_enc_vectors_list = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

    def sum_enc_all_reduce(self, request, context):
        server_start = time.time()
        client_rank = request.client_rank
        print(">>> All reduce server receive encrypted data from client {}, time = {} ----"
              .format(client_rank, time.asctime(time.localtime(time.time()))))
        # deserialize vector from bytes
        deser_start = time.time()
        enc_vector = ts.ckks_vector_from(self.ctx, request.msg)
        deser_time = time.time() - deser_start
        self.sum_enc_vectors_list.append(enc_vector)
        self.n_sum_request += 1
        # wait until receiving of all clients' requests
        print("Number of Sum Request: ", self.n_sum_request)
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start
        if client_rank == 0:
            sum_start = time.time()
            summed_enc_vector = sum(self.sum_enc_vectors_list)
            self.sum_data.append(summed_enc_vector)
            sum_time = time.time() - sum_start
            self.n_sum_round = 0
            print("Sum Time: ", sum_time)
            self.sum_completed = True
        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start
        response_start = time.time()
        response = tenseal_shapley_data_pb2.all_reduce_msg(
            client_rank=client_rank,
            msg=self.sum_data[0].serialize()
        )
        response_time = time.time() - response_start
        self.n_sum_response = self.n_sum_response + 1
        print("Number of Sum Response: ", self.n_sum_response)
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)
        if client_rank == 0:
            self.reset_sum()
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)
        print("Number of Sum Round: ", self.n_sum_round)
        print(">>> server finish sum_enc, cost {:.2f} s: deserialization {:.2f} s, "
              "wait for requests {:.2f} s, wait for sum {:.2f} s, create response {:.2f} s"
              .format(time.time() - server_start, deser_time,
                      wait_time, sum_wait_time, response_time))
        return response

    def sum_enc_batch(self, request, context):
        server_start = time.time()
        client_rank = request.client_rank
        print(">>> Batch server receive encrypted data from client {}, time = {} ----"
              .format(client_rank, time.asctime(time.localtime(time.time()))))
        # deserialize vector from bytes
        deser_start = time.time()
        enc_vector = ts.ckks_vector_from(self.ctx, request.msg)
        deser_time = time.time() - deser_start
        self.sum_enc_vectors_dict[client_rank] = enc_vector
        self.n_sum_request += 1
        # wait until receiving of all clients' requests
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start
        if client_rank == 0:
            sum_start = time.time()
            group_keys = request.group_keys
            print(group_keys)
            summed_enc_vector = sum_as_group(self.sum_enc_vectors_dict, group_keys, self.num_clients)
            serialized_enc_vector = [vector.serialize() for vector in summed_enc_vector]
            self.sum_data.append(serialized_enc_vector)
            sum_time = time.time() - sum_start
            self.n_sum_round = 0
            self.sum_completed = True
        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start
        response_start = time.time()
        response = tenseal_shapley_data_pb2.batch_msg(
            client_rank=client_rank,
            res=self.sum_data[0]
        )
        response_time = time.time() - response_start
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)
        if client_rank == 0:
            self.reset_sum()
        self.n_sum_round = self.n_sum_round + 1
        print("Number of Sum Round: ", self.n_sum_round)
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)
        print(">>> server finish sum_enc, cost {:.2f} s: deserialization {:.2f} s, "
              "wait for requests {:.2f} s, wait for sum {:.2f} s, create response {:.2f} s"
              .format(time.time() - server_start, deser_time,
                      wait_time, sum_wait_time, response_time))
        return response



def sum_as_group(enc_vectors, group_keys, num_clients):
    group_enc_results = []
    for key in group_keys:
        group_flags = utility_key_to_groups(key, num_clients)
        enc_vector = sum([enc_vectors[i] for i in range(len(enc_vectors)) if group_flags[i]])
        group_enc_results.append(enc_vector)
    return group_enc_results

def launch_server(address, num_clients, ctx_file):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = ShapleyServer(address, num_clients, ctx_file)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    tenseal_shapley_data_pb2_grpc.add_MIServiceServicer_to_server(servicer, server)

    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = global_args_parser()
    server_address = args.a_server_address
    # num_clients = args.num_clients
    num_clients = args.world_size
    ctx_file = args.config
    launch_server(server_address, num_clients, ctx_file)
