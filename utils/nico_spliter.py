import os


# used to remove all .DS_Store files from mac os.
def file_fliter(root_path):
    for home, dirs, files in os.walk(root_path):
        for file_name in files:
            if file_name.startswith("."):
                print(os.path.join(home, file_name))
                try:
                    os.remove(os.path.join(home, file_name))
                except:
                    print('Wrong path...')


def delete_blank(root_path):
    for category in os.listdir(root_path):
        for domain in os.listdir(os.path.join(root_path, category)):
            if "on grass" in domain:
                os.rename(os.path.join(root_path, category, "on grass"), os.path.join(root_path, category, "grass"))
            if "on snow" in domain:
                os.rename(os.path.join(root_path, category, "on snow"), os.path.join(root_path, category, "snow"))


def fill_blank(root_path):
    for home, dirs, files in os.walk(root_path):
        for file_name in files:
            if " " in file_name:
                print(os.path.join(home, file_name))
                try:
                    os.rename(os.path.join(home, file_name), os.path.join(home, file_name.replace(" ", '_')))
                    # os.remove(os.path.join(home, file_name))
                except:
                    print('Wrong path...')


def read_dataset(root_path):
    ds = {}
    domain_freq = {}
    for category in os.listdir(root_path):
        ds[category] = {}
        # new a dict to store each category's domain information.
        in_cate = {}
        for domain in os.listdir(os.path.join(root_path, category)):
            if domain not in domain_freq:
                domain_freq[domain] = 0
            domain_freq[domain] += 1
            # print(domain)
            in_cate[domain] = []
            for file in os.listdir(os.path.join(root_path, category, domain)):
                in_cate[domain].append(os.path.join(category, domain, file))
                # print(in_cate[domain][-1])
            ds[category] = in_cate

    return ds


def domain_spliter(ds, source_domain, target_domain, pct_lists):
    source = []
    for category in ds:
        sub_area_images = ds[category][source_domain]
        sub_area_image_ext = []
        for item in sub_area_images:
            sub_area_image_ext.append((item, category))
        source.extend(sub_area_image_ext)

    pass
    import numpy as np
    prepared_target = {}

    def select_by_category(domain, pct_list: list):
        labeled = {}
        for n_pct in pct_list:
            labeled[n_pct] = []
        unlabed = []
        val = []

        def rp_list(arr, cat):
            ret_arr = []
            for index, item in enumerate(arr):
                ret_arr.append((item, cat))
            return ret_arr

        for category in ds:
            domain_arr = ds[category][domain]
            np.random.shuffle(domain_arr)
            for n_pct in pct_list:
                if int(len(ds[category][domain]) * 0.01 * n_pct) >= 1:
                    selection = int(len(ds[category][domain]) * 0.01 * n_pct)
                else:
                    selection = 1
                cat_labeled_train = domain_arr[:selection]
                labeled[n_pct].extend(rp_list(cat_labeled_train, category))

            cat_unlabeled_train = domain_arr[:int(len(domain_arr) * 0.7)]
            cat_val = domain_arr[int(len(domain_arr) * 0.7):]

            unlabed.extend(rp_list(cat_unlabeled_train, category))
            val.extend(rp_list(cat_val, category))

        return labeled, unlabed, val

    prepared_target = {}
    prepared_target['labeled'], prepared_target['unlabeled'], prepared_target[
        'validation'] = select_by_category(target_domain, pct_lists)

    return source, prepared_target


def write_to_txt(source: list, save_path: str, save_name: str):
    def refiner(line: str):
        if 'on snow' in line:
            line = line.replace('on snow', 'snow')
        if 'on grass' in line:
            line = line.replace('on grass', 'grass')
        return line

    with open(os.path.join(save_path, save_name), 'w') as fp:
        for item in source:
            fp.writelines(refiner(item[0]) + ' ' + item[1] + '\n')


if __name__ == '__main__':
    # file_fliter("/home/v-boli4/codebases/external_datasets/NICO-Traffic")
    fill_blank("/home/v-boli4/codebases/external_datasets/NICO-ANIMAL")
    ds = read_dataset('/home/v-boli4/codebases/external_datasets/NICO-ANIMAL')
    ds.pop('bear', None)
    ds.pop('bird', None)

    pct_lists = [1, 3, 5, 10]


    def pipeline(source_domain, target_domain):
        source, prepared_target = domain_spliter(ds, source_domain, target_domain, pct_lists)
        write_to_txt(source, '/home/v-boli4/codebases/DA_Codebase/datasets/convention/nico/source',
                     '{}.txt'.format(source_domain))
        for pct in pct_lists:
            write_to_txt(prepared_target['labeled'][pct],
                         '/home/v-boli4/codebases/DA_Codebase/datasets/convention/nico/target',
                         '{}_labeled_{}.txt'.format(target_domain, pct))
        write_to_txt(prepared_target['unlabeled'],
                     '/home/v-boli4/codebases/DA_Codebase/datasets/convention/nico/target',
                     '{}_{}.txt'.format(target_domain, 'unlabeled'))
        write_to_txt(prepared_target['validation'],
                     '/home/v-boli4/codebases/DA_Codebase/datasets/convention/nico/target',
                     '{}_{}.txt'.format(target_domain, 'validation'))


    pipeline('grass', 'snow')
    pipeline('snow', 'grass')
