import os
import torch
from dialogue.models.constructor import construct_model
from dialogue.toolbox.stats import Statistics


class Trainer(object):
    def __init__(self, train_iter, valid_iter,
                 vocabs, optimizer, train_opt, logger):
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.vocabs = vocabs
        self.optimizer = optimizer
        self.train_opt = train_opt
        self.model = construct_model(train_opt, vocabs["pre_word_emb"])
        if train_opt.meta.use_cuda:
            self.model = self.model.cuda()
        self.logger = logger

        self.optimizer.set_parameters(self.model.named_parameters())
        self.best_score = float('inf')
        self.step = 0

    def train(self):
        total_stats = Statistics(self.logger)
        report_stats = Statistics(self.logger)
        for batch in self.train_iter:
            self.model.zero_grad()
            result_dict = self.model.run_batch(batch)
            loss = result_dict["loss"]
            loss.div(self.train_opt.meta.batch_size).backward()
            self.optimizer.step()

            batch_stats = Statistics(num=result_dict["num_words"],
                                     loss=loss.item(),
                                     n_words=result_dict["num_words"],
                                     n_correct=result_dict["num_correct"],
                                     logger=self.logger)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            if self.step and self.step % self.train_opt.meta.print_every == 0:
                report_stats.output(self.step, self.train_opt.meta.total_steps)
            if self.step and self.step % self.train_opt.meta.valid_every == 0:
                self.hit_checkpoint(total_stats)
            if self.step > self.train_opt.meta.total_steps:
                break
            self.step += 1

    def _validate(self):
        self.model.eval()
        self.model.flatten_parameters()
        stats = Statistics(self.logger)
        for j, batch in enumerate(self.valid_iter):
            result_dict = self.model.run_batch(batch)
            batch_stats = Statistics(num=result_dict["num_words"],
                                     loss=result_dict["loss"].item(),
                                     n_words=result_dict["num_words"],
                                     n_correct=result_dict["num_correct"],
                                     logger=self.logger)
            stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()
        return stats

    def hit_checkpoint(self, train_stats):
        self.logger.info('Train loss: %g' % train_stats.get_loss())
        self.logger.info('Train perplexity: %g' % train_stats.ppl())
        self.logger.info('Train accuracy: %g' % train_stats.accuracy())

        valid_stats = self._validate()
        self.logger.info('Valid loss: %g' % valid_stats.get_loss())
        self.logger.info('Valid perplexity: %g' % valid_stats.ppl())
        self.logger.info('Valid accuracy: %g' % valid_stats.accuracy())
        if valid_stats.ppl() < self.best_score:
            self.best_score = valid_stats.ppl()
            self.save_checkpoint(valid_stats.ppl(), "best_model.pt")
            self.logger.info("Save best model..")
        self.logger.info("Learning rate: {}".format(self.optimizer.learning_rate))

    def save_checkpoint(self, score_dict, ckp_name):
        model_file = {
            "saved_step": self.step,
            "model": self.model,
            "score": score_dict,
            "train_opt": self.train_opt,
            "vocabs": self.vocabs,
        }
        torch.save(model_file, os.path.join(
            self.train_opt.meta.save_model, ckp_name))