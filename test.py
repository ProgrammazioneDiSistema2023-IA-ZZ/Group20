import onnx_binding as onnx

print('creating service')
service = onnx.build_service('onnx-runtime/tests/models/resnet50-v2-7.onnx', 4)
print('created')
paths = ['onnx-runtime/tests/images/apples_on_table.jpeg']
print('calling prepare_and_run')
output = service.prepare_and_run(paths)
print('done')
print('getting top 5 labels')
top_ks = output.get_top_k_predictions(5)
for top_k in top_ks:
    print('Top K of ')
    for pred in top_k:
        print(f'{pred}')