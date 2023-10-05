from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import List
import pydantic
from random import randint
import string
from random import choice
import os

class Generator(pydantic.BaseModel):
    config: dict
    
    @staticmethod
    def generate_random_string(length=8):
        return ''.join(choice(string.ascii_letters + string.digits) for _ in range(length))

    
    @staticmethod
    def _from_template(template_file: Path, config: dict) -> str:
        template_dir: Path = os.path.dirname(os.path.abspath(template_file))
        env = Environment(loader=FileSystemLoader(template_dir))
        template: Template = env.get_template(os.path.basename(template_file))
        config["paramters"]["shutdown"] = randint(config["paramters"]["shutdown"],2*config["paramters"]["shutdown"])
        return template.render(**config)
    
    @staticmethod
    def _write_to_file(out: Path, content: str) -> Path:
        random_name = Generator.generate_random_string()
        output_file = out / f"output_{random_name}.yaml"
        with open(output_file, "w") as file:
            file.write(content)
        return output_file
    
    def _generate(self, template_file: Path, number: int = 10) -> None:
        for _ in range(number):
            yield self._from_template(template_file=template_file,config=self.config)
    
    
    def create(self, template_file: Path, out: Path ,number: int = 10) -> List[str]:
        genreator = self._generate(template_file=template_file,number=number)
        out.mkdir(parents=True, exist_ok=True)
        files_names: list = []
        for content in genreator:
            name = self._write_to_file(out=out,content=content)
            files_names.append(str(name))
        return files_names
            