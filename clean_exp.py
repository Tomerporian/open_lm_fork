import os
import re
def clean_exp(dir):
    # for all subdirectories in dir
    for subdir in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, subdir)):
            # remove 'checkpoints' directory in subdir
            if os.path.exists(os.path.join(dir, subdir, 'checkpoints')):
                os.system('rm -r ' + os.path.join(dir, subdir, 'checkpoints'))
                # print('rm -r ' + os.path.join(dir, subdir, 'checkpoints'))
                # print(dir)

def check_nccl(dir):
    # for all subdirectories in dir
    for subdir in os.listdir(dir):
        # dubdirectory should be a directory, and contain files with <job_id>.out
        if os.path.isdir(os.path.join(dir, subdir)) and "done" not in os.listdir(os.path.join(dir, subdir)):
            files = os.listdir(os.path.join(dir, subdir))
            # we want to check to file with max job_id
            max_job_id = -1
            for file in files:
                if '.out' in file:
                    job_id = int(file.split('.')[0])
                    if job_id > max_job_id:
                        max_job_id = job_id
            file_to_check = str(max_job_id) + '.out'
            # check if file_to_check contains 'To avoid data inconsistency, we are taking the entire process down.'
            with open(os.path.join(dir, subdir, file_to_check), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'To avoid data inconsistency, we are taking the entire process down.' in line:
                        print(os.path.join(dir, subdir))
                        break

def main():
    dir_name = 'exps_final_runs'
    dirs_to_keep = ['24-05-11-kaplan_1_nodes_big_BS', '24-05-11-kaplan_2_nodes_big_BS', '24-05-11-kaplan_4_nodes_big_BS', '24-05-10-kaplan_8_nodes_orig_fix', '24-05-14-CC_1_nodes_rw_BS_256_rerun', '24-05-14-CC_2_nodes_rw_BS_256_rerun', '24-05-14-CC_4_nodes_rw_BS_256_rerun', '24-05-14-CC_8_nodes_rw_BS_256_rerun', '24-05-09-const_1_nodes_tuned', '24-05-09-const_2_nodes_tuned', '24-05-09-const_4_nodes_tuned', '24-05-09-const_8_nodes_tuned', '24-05-12-kaplan_1_nodes_big_BS_openwebtext2', '24-05-12-kaplan_2_nodes_big_BS_openwebtext2', '24-05-12-kaplan_4_nodes_big_BS_openwebtext2', '24-05-11-kaplan_8_nodes_openwebtext2', '24-05-14-CC_1_nodes_owt2_BS_256_rerun', '24-05-14-CC_2_nodes_owt2_BS_256_rerun', '24-05-14-CC_4_nodes_owt2_BS_256_rerun', '24-05-14-CC_8_nodes_owt2_BS_256_rerun', '24-05-11-const_1_nodes_tuned_openwebtext2', '24-05-11-const_2_nodes_tuned_openwebtext2', '24-05-11-const_4_nodes_tuned_openwebtext2', '24-05-11-const_8_nodes_tuned_openwebtext2', '24-04-28-final_sweep_const', '24-04-29-final_sweep_const_extra_1', '24-04-29-final_sweep_const_extra_2', '24-05-01-final_sweep_const_extra_3', '24-05-02-final_sweep_const_extra_4', '24-05-03-final_sweep_const_beta2', '24-05-03-final_sweep_const_beta2_220M', '24-05-05-final_sweep_const_extra_5', '24-05-17-seed_variance_rerun', '24-05-20-seed_variance_owt']

    for dir in sorted(os.listdir(dir_name)):
        if dir in dirs_to_keep:
            continue
        else:
            clean_exp(os.path.join(dir_name, dir))
            # print(dir)
    # dirs_to_check = [dir for dir in os.listdir('exps_final_runs')]
    # for dir in dirs_to_check:
    #     check_nccl(os.path.join('exps_final_runs', dir))
if __name__ == '__main__':
    main()