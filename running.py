from torch.utils.data import DataLoader
# Import Project Modules ---------------------------------------------------------
from utils import dataset_class
from Models.model import Encoder_factory, count_parameters
from Models.loss import get_loss_module
from Models.utils import load_model
from trainer import *
# --------- For Logistic Regression--------------------------------------------------
from eval import fit_lr,  make_representation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


####
logger = logging.getLogger('__main__')


def Rep_Learning(config, Data):
    # ---------------------------------------- Self Supervised Data -------------------------------------
    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config['patch_size'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # For Linear Probing During the Pre-Training
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    logger.info("Pre-Training Self Supervised model ...")
    config['Data_shape'] = Data['All_train_data'].shape
    config['num_labels'] = int(max(Data['All_train_label'])) + 1
    Encoder = Encoder_factory(config)

    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    # Specify which networks you want to optimize
    networks_to_optimize = [Encoder.contex_encoder, Encoder.InputEmbedding, Encoder.Predictor]
    # networks_to_optimize = [Encoder.contex_encoder, Encoder.Predictor]
    # networks_not_to_optimize = [Encoder.target_encoder, Encoder.PatchEmbedding]
    # Convert parameters to tensors
    params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
    params_not_to_optimize = [p for p in Encoder.target_encoder.parameters()]
    # params_not_to_optimize = [p for net in networks_not_to_optimize for p in net.parameters()]

    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class([{'params': params_to_optimize, 'lr': config['lr']},
                                       {'params': params_not_to_optimize, 'lr': 0.0}])

    config['problem_type'] = 'Self-Supervised'
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))
    # tensorboard_writer = SummaryWriter('summary')
    Encoder.to(config['device'])
    # ------------------------------------------------- Training The Model ---------------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = Self_Supervised_Trainer(Encoder, train_loader, test_loader, config, l2_reg=0, print_conf_mat=False)
    SS_train_runner(config, Encoder, SS_trainer, save_path)
    # **************************************************************************************************************** #
    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    train_repr, train_labels = make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = make_representation(SS_Encoder, test_loader)

    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    plot_tSNE(test_repr.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)
    # print("Test ROC AUC:")
    # print(roc_auc_score(y_hat, test_labels.cpu().detach().numpy()))

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config['patch_size'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config['patch_size'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    logger.info('Starting Fine_Tuning...')
    S_trainer = SupervisedTrainer(SS_Encoder, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(SS_Encoder, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    Strain_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics


def Supervised(config, Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    config['Data_shape'] = Data['train_data'].shape
    config['num_labels'] = int(max(Data['train_label'])) + 1
    Encoder = Encoder_factory(config)

    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)

    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))
    # tensorboard_writer = SummaryWriter('summary')
    Encoder.to(config['device'])
    # ------------------------------------------------- Training The Model ---------------------------------------------

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config['patch_size'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config['patch_size'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = SupervisedTrainer(Encoder, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(Encoder, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    Strain_runner(config, Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics


def plot_tSNE(data, labels):
    # Create a TSNE instance with 2 components (dimensions)
    tsne = TSNE(n_components=2, random_state=42)
    # Fit and transform the data using t-SNE
    embedded_data = tsne.fit_transform(data)

    # Separate data points for each class
    class_0_data = embedded_data[labels == 0]
    class_1_data = embedded_data[labels == 1]

    # Plot with plt.plot
    plt.figure(figsize=(6, 5))  # Set background color to white
    plt.plot(class_0_data[:, 0], class_0_data[:, 1], 'bo', label='Real')
    plt.plot(class_1_data[:, 0], class_1_data[:, 1], 'ro', label='Fake')
    plt.legend(fontsize='large')
    plt.grid(False)  # Remove grid
    plt.savefig('SSL.pdf', bbox_inches='tight', format='pdf')
    # plt.show()