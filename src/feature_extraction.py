import os
from copy import deepcopy
import csv
import stat
import shutil
from glob import glob
import json
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch
from jarvis.core.atoms import Atoms
from tqdm import tqdm


def copy_with_write_permission(src, dst):
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IWUSR)


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="directory containing config.json and best_model.pt")
    parser.add_argument("--save", default="./features/", help="directory to save features and labels")
    parser.add_argument("--alignn_training_dir", required=True, help="alignn training dir")


    args = parser.parse_args()
    save_dir = args.save
    if not os.path.exists(save_dir):
        os.mkdir(f"{save_dir}/")
    data_dir = args.data
    alignn_trainig_dir = args.alignn_training_dir
    if not data_dir:
        data_dir = list(glob("./*/*/"))[0]
    shutil.copytree(data_dir, f"{save_dir}/tmp_model", copy_function=copy_with_write_permission) # avoid ids_train_val_test.json and *_data_range are overwritten.

    config = loadjson(os.path.join(save_dir, "tmp_model", "config.json"))
    _config = ALIGNNAtomWiseConfig(**config["model"])

    # data
    id_prop_csv = os.path.join(alignn_trainig_dir, "id_prop.csv")
    id_prop_csv_file = True
    with open(id_prop_csv, "r") as f:
        reader = csv.reader(f)
        dat = [row for row in reader]
    print("id_prop_csv_file exists", id_prop_csv_file)

    n_outputs = []
    dataset = []
    reference_data = dict()
    for i in dat:
        info = {}
        if id_prop_csv_file:
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]
            info["jid"] = file_name

            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
                n_outputs.append(tmp)
            info["target"] = deepcopy(tmp)
            reference_data[file_name] = tmp
            file_path = os.path.join(alignn_trainig_dir, file_name)
            atoms = Atoms.from_poscar(file_path)
            info["atoms"] = atoms.to_dict()
        dataset.append(info)
    print("len dataset", len(dataset))
    del dat

    config = TrainingConfig(**config)
    (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ) = get_train_val_loaders(
            dataset_array=dataset,
            target="target",
            target_atomwise=None,
            target_grad=None,
            target_stress=None,
            train_ratio=1,
            val_ratio=0,
            test_ratio=0,
            line_graph=True,
            batch_size=1,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=os.path.join(save_dir, "tmp_model"),
            cutoff=config.cutoff,
            cutoff_extra=config.cutoff_extra,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=os.path.join(save_dir, "tmp_model"),
            split_seed=config.random_seed,
        )

    # load model
    model = ALIGNNAtomWise(config=_config)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'tmp_model', 'best_model.pt'), map_location=torch.device('cpu')))
    model.eval()
    print("model loading")

    net = model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device)

    data = []
    for loader in [val_loader]: # , val_loader, test_loader]:
        for dats, jid in tqdm(list(zip(loader, loader.dataset.ids))):
            tmp = {}
            tmp["id"] = jid
            tmp["reference"] = reference_data[jid]
            graph_list = []
            alignn_features = []
            gcn_features = []
            fc_features = []
            alignn_hooks = []
            gcn_hooks = []
            fc_hooks = []


            def alignn_hook(module, input, output):
                alignn_features.append(output)

            def gcn_hook(module, input, output):
                gcn_features.append(output)

            def fc_hook(module, input, output):
                fc_features.append(output)

            for _, alignn_layer in enumerate(model.alignn_layers):
                hook_handle = alignn_layer.register_forward_hook(alignn_hook)
                alignn_hooks.append(hook_handle)

            for _, gcn_layer in enumerate(model.gcn_layers):
                hook_handle = gcn_layer.register_forward_hook(gcn_hook)
                gcn_hooks.append(hook_handle)

            hook_handle = model.readout.register_forward_hook(fc_hook)
            fc_hooks.append(hook_handle)

            with torch.no_grad():
                result = net([dats[0].to(device), dats[1].to(device)])

                # temp
                if isinstance(result, dict) and "out" in result:
                    tmp["prediction"] = result["out"].cpu().tolist()
                else:
                    tmp["prediction"] = result.cpu().tolist()

                for idx, feature in enumerate(alignn_features):
                    atom_features = feature[0].mean(dim=0)
                    tmp[f"ALIGNN{idx+1}"] = atom_features.tolist()

                for idx, feature in enumerate(gcn_features):
                    atom_features = feature[0].mean(dim=0)
                    tmp[f"CGN{idx+1}"] = atom_features.tolist()

                for idx, feature in enumerate(fc_features):
                    tmp[f"Last{idx+1}"] = torch.squeeze(feature).tolist()

                for hook_handle in alignn_hooks:
                    hook_handle.remove()
                for hook_handle in gcn_hooks:
                    hook_handle.remove()
                for hook_handle in fc_hooks:
                    hook_handle.remove()
            data.append(tmp)

    print(f"data len: {len(data)}")

    with open(os.path.join(save_dir, 'features.json'), 'w') as f:
      json.dump(data, f)


if __name__ == "__main__":
    main()
