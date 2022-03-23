import os
import json
import pickle


class cdmss:
    def __init__(self, argv):
        self.parameters = argv[1]  # 超参数列表 json格式
        self.info_path = argv[2]   # 输入信息与输出结果保存地址 json格式
        self.info_dict = self.read_json(self.info_path)  # 与输出相关的信息


        self.parameters_dict = self.read_json(self.parameters)  # 超参数信息
        self.datafile = self.info_dict['Data_Path']        # 数据集地址

        self.model_name = self.info_dict['Model_Name']  # 本次任务所使用或保存的模型名：算法名+唯一标识串（eg. randomforest1640917882844）
        self.task_id = self.info_dict['Task_ID']  # 本次任务标识：算法名+唯一标识串（eg. randomforest636364555）
        self.model_save_directory = self.info_dict['Model_Save_Dir']  # 模型参数保存路径
        self.predict_directory = self.info_dict["Predict_Dir"]  # 预测结果（csv）保存路径
        self.img_directory = self.info_dict["Img_Dir"]  # 预测图片（image）保存路径
        self.report_directory = self.info_dict["Report_Dir"]  # 预测评价指标（json）保存路径
        self.message_directory = self.info_dict["Message_Dir"]  # 运行信息（json）保存路径
        self.Output_n = self.info_dict["Output_n"]  # 输出维度

        self.model_save_path = os.path.join(self.model_save_directory, self.model_name + '_model.pkl')   # 模型(model_name)
        self.report_path = os.path.join(self.report_directory, self.task_id + '_report.json')            # 评价指标（task_id）
        self.predict_path = os.path.join(self.predict_directory, self.task_id + '_predict.csv')          # 预测结果（task_id）
        self.message_path = os.path.join(self.message_directory, self.task_id + '_message.json')         # 运行信息（task_id）



    def get_data(self, output_type='df', encoding='utf-8'):
        """
        输出方式：默认'df'
        'df': (input_dataframe, output_dataframe)
        'np': (np.array(header), np.array(input_content), np.array(output_content), )
        'list': ([header], [content], [output])
        """
        if output_type == 'df':
            import pandas as pd
            data = pd.read_csv(self.datafile, encoding=encoding)
            data_x = data[data.columns[0: -self.Output_n]]
            data_y = data[data.columns[-self.Output_n]]
            return data_x, data_y

        elif output_type == 'np':
            import numpy as np
            with open(self.datafile, encoding=encoding) as f:
                data = np.loadtxt(f, str, delimiter=",")
            try:
                content = data[1:].astype(np.float32)
            except:
                content = data[1:]
            return data[0], content[:, 0: -self.Output_n], content[:, -self.Output_n]

        elif output_type == 'list':
            import csv
            data = []
            with open(self.datafile) as csvfile:
                csv_reader = csv.reader(csvfile)
                header = next(csv_reader)
                for row in csv_reader:
                    data.append(row)
            return header, [x[0: -self.Output_n] for x in data], [x[-self.Output_n] for x in data]

    def get_parameters(self):
        return self.parameters_dict

    def save_model(self, model):
        if not os.path.exists(self.model_save_directory):
            os.makedirs(self.model_save_directory)

        s = pickle.dumps(model)
        with open(self.model_save_path, 'wb+') as f:
            f.write(s)
        return self.model_save_path

    def get_model(self):
        f = open(self.model_save_path, 'rb')  # 注意此处model是rb
        s = f.read()
        return pickle.loads(s)

    def save_result(self, dic):
        # 处理图像
        if 'img' in dic and 'image_type' in dic and len(dic['img']) == len(dic['image_type']):
            if not os.path.exists(self.img_directory):
                os.makedirs(self.img_directory)

            for i, e in enumerate(dic['img']):
                img_path = os.path.join(self.img_directory, self.task_id + '_img_' + str(i) + '.png')
                if dic['image_type'][i] == 'plt':
                    e.savefig(img_path)
                elif dic['image_type'][i] == 'pil_img':
                    e.save(img_path)
                elif dic['image_type'][i] == 'np_img':
                    from PIL import Image
                    im = Image.fromarray(e)
                    im.save(img_path)

        # 处理report 评价指标
        if 'report' in dic:
            if not os.path.exists(self.report_directory):
                os.makedirs(self.report_directory)
            with open(self.report_path, "w") as f:
                json.dump(dic['report'], f)

        # 处理预测结果
        if 'result' in dic and 'result_type' in dic:
            if not os.path.exists(self.predict_directory):
                os.makedirs(self.predict_directory)

            if dic['result_type'] == 'df':
                import pandas as pd
                label_df = self.get_data(output_type='df')[1]
                result_df = pd.concat([label_df, dic['result']], axis=1)
                result_df.columns = ['Label' for x in range(int(result_df.shape[1]/2))] + ['Result' for x in range(int(result_df.shape[1]/2))]
                result_df.to_csv(self.predict_path, index=False)

            elif dic['result_type'] == 'np':
                import numpy as np
                label_np = self.get_data(output_type='np')[2].reshape(-1, self.Output_n)
                result_np = np.concatenate((label_np, dic['result']), axis=1)
                import csv
                with open(self.predict_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Label' for x in range(int(result_np.shape[1]/2))] + ['Result' for x in range(int(result_np.shape[1]/2))])
                    writer.writerows(result_np)

            elif dic['result_type'] == 'list':
                import numpy as np
                label_np = self.get_data(output_type='np')[2].reshape(-1, self.Output_n)
                result_np = np.concatenate((label_np, np.array(dic['result']).reshape(-1, self.Output_n)), axis=1)
                # np.savetxt(self.predict_path, result_np, delimiter=', ')
                import csv
                with open(self.predict_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Label' for x in range(int(result_np.shape[1]/2))] + ['Result' for x in range(int(result_np.shape[1]/2))])
                    writer.writerows(result_np)

    def save_message(self, code, message):
        if not os.path.exists(self.message_directory):
            os.makedirs(self.message_directory)
        with open(self.message_path, "w") as f:
            json.dump({"code": code, "message": message}, f)

    def read_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            str = file.read()
            data_dict = json.loads(str)
        return data_dict


if __name__ == "__main__":
    cdmss = cdmss([0,
                   r'Z:\USTB\北科大\0研究生\20211207腐蚀算法大系统Java\python模板\param.json',
                   r'Z:\USTB\北科大\0研究生\20211207腐蚀算法大系统Java\python模板\exampleWrite.json'])
    # print(cdmss.get_data('list'))
    # print(cdmss.save_model('list'))
