import onnx_binding as onnx

def run(image_path, network_path, k=5):
    service = onnx.build_service(network_path, 4)
    paths = [image_path]
    output = service.prepare_and_run(paths)
    top_ks = output.get_top_k_predictions(k)

    prediction=""
    sum=0
    for top_k in top_ks:
        for pred in top_k:
            if sum > 95:
                break
            str_pred = str(pred).replace('(', '').replace(')', '')
            perc = str_pred.split(',')[-1]
            perc_val = perc.replace('%', '')
            sum+=float(perc_val)
            
            prediction+=str_pred+"\n"

    return prediction