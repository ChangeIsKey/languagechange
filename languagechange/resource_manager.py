import json
import os
import urllib.request
from pathlib import Path
from platformdirs import user_cache_dir
import dload
import logging

# Configure logging with a basic setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LanguageChange():

    def __init__(self):
        self.cache_dir = user_cache_dir("languagechange", "Change is Key!")
        self.resources_dir = os.path.join(self.cache_dir, 'resources')
        self.models_dir = os.path.join(self.cache_dir, 'models')
        self.local_dir = os.path.join(self.cache_dir, 'local')
        self.local_resources_dir = os.path.join(self.local_dir, 'resources')
        self.local_models_dir = os.path.join(self.local_dir, 'models')
        Path(self.resources_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        self.load_resources_hub()

    def load_resources_hub(self):
        with urllib.request.urlopen('https://raw.githubusercontent.com/pierluigic/languagechange/main/languagechange/resources_hub.json') as url:
            self.resource_hub = json.load(url)

    def download_ui(self):
        j = 0
        list_resources = []

        logger.info('Available resources:\n')

        for resource_type in self.resource_hub:
            logger.info('########################')
            logger.info('###### '+resource_type+' ######')
            logger.info('########################\n')
            for resource_name in self.resource_hub[resource_type]:
                logger.info(resource_name)
                logger.info('---------------------')
                for dataset in self.resource_hub[resource_type][resource_name]:
                    logger.info('\t -'+dataset)
                    for version in self.resource_hub[resource_type][resource_name][dataset]:
                        logger.info(f'\t\t{j}) '+version)
                        list_resources.append([resource_type,resource_name,dataset,version])
                        j = j + 1
                logger.info('\n')

        findchoice = False

        while not findchoice:
            choice = input(f'Select an option (0-{j}), digit -1 to exit: ')
            try:
                choice = int(choice.strip())
                if choice >= -1 and choice <= j:
                    findchoice = True
                else:
                    logger.info(f'Only numbers in the range (0-{j}) are allowed, digit -1 to exit.')
            except:
                logger.error(f'Only numbers in the range (0-{j}) are allowed, digit -1 to exit.')

        if not choice == -1:

            options = {'yes':1,'y':1,'no':0,'n':0}
            confirm = ""

            while not confirm.strip().lower() in {'yes','y','no','n'}:
                choice_resource = '/'.join(list_resources[choice])
                confirm = input(f'You have choice {choice} ({choice_resource}), do you confirm your choice? (yes/y/no/n): ')
            
            confirm = options[confirm]
            if confirm:
                logger.info('Downloading the required resource...')
                self.download(*list_resources[choice])
                logger.info('Completed!')
            else:
                self.download_ui()

    def download(self, resource_type, resource_name, dataset, version):
        try:
            url = self.resource_hub[resource_type][resource_name][dataset][version]
            destination_path = os.path.join(self.resources_dir,resource_type,resource_name,dataset,version)
            Path(destination_path).mkdir(parents=True, exist_ok=True)
            dload.save_unzip(url, destination_path, delete_after=True)
            return os.path.join(self.resources_dir,resource_type,resource_name,dataset,version)
        except:
            logger.error('ERROR: Cannot download the resource.')
            return None

    def get_resource(self, resource_type, resource_name, dataset, version):
        path = os.path.join(self.resources_dir,resource_type,resource_name,dataset,version)
        if os.path.exists(path):
            return path
        else:
            result = self.download(resource_type, resource_name, dataset, version)
            return result

    def save_resource(self, resource_type, resource_name, dataset, version):
        path = os.path.join(self.local_resources_dir,resource_type,resource_name,dataset,version)
        Path(path).mkdir(parents=True, exist_ok=True)
        return path