import requests
import os
import zipfile
import gzip
import shutil
from pathlib import Path
import time
import copy


API_KEY = "ZiTGtE4ATfq5v9vAJSct9n6rQPgLXur5RfoKsDRp"
EMAIL = "363483835@qq.com"
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download.json?"

# 多州数据配置
POINTS_BY_STATE = {
    # Cfa 亚热带湿润性气候 humid subtropical climate
    # 'Oklahoma': [
    #     ('2751838', 'Oklahoma City'),
    #     ('2753970', 'Moore'),
    #     ('2758210', 'Norman'),
    #     ('2727601', 'Mustang'),
    #     ('2756080', 'Edmond'),
    #     ('2776956', 'Choctaw'),
    #     ('2774883', 'Jones'),
    #     ('2723172', 'Yukon')
    # ],
    # Csb 暖夏地中海气候 warm-summer Mediterranean climate
    # 'Oregon': [
    #     ('542246', 'Wilsonville'),
    #     ('550135', 'West Linn'),
    #     ('552415', 'Oregon City'),
    #     ('543358', 'Tualatin'),
    #     ('547860', 'Lake Oswego'),
    #     ('542239', 'Tigrad'),
    #     ('547855', 'Portland'),
    #     ('548995', 'Milwaukie'),
    # ],
    # Bsk 寒冷半干旱气候 cold semi-arid climate
    # 'Texas': [
    #     ('2280292', 'Lubbock'),
    #     ('2265447', 'Shallowater'),
    #     ('2297304', 'Idalou'),
    #     ('2263342', 'Wolfforth'),
    #     ('2301573', 'Slaton'),
    #     ('2225286', 'Levelland'),
    #     ('2235876', 'Brownfield'),
    #     ('2286689', 'Tahoka')
    # ],

    # 'California' : [
    #     ('830655', 'Los Angeles'),
    #     ('904327', 'San Diego'),
    #     ('948428', 'Palm Springs'),
    #     ('563176', 'San Francisco'),
    #     ('726962', 'Fresno'),
    #     ('617994', 'Sacramento'),
    #     ('778706', 'Bakersfield'),
    #     ('593824', 'San Jose')
    # ],

    # Dfa（热夏湿润大陆性气候）
    'Illinois_cities': [
        ('3409513', 'Canton'),
        ('3506677', 'Bloomington'),
        ('3525065', 'Lexington'),
        ('3532336', 'Chenoa'),
        ('3562878', 'Gibson City'),
        ('3577145', 'Urbana'),
        ('3582437', 'Rantoul'),
        ('3587716', 'Paxton'),

    ],

    # Csa（热夏地中海气候）
    'California_cities' : [
        ('813784', 'Santa Monica'),
        ('826430', 'Burbank'),
        ('830650', 'Glendale'),
        ('830655', 'Los_Angeles'),
        ('837660', 'Pasadena'),
        ('845919', 'El Monte'),
        ('852588', 'West Covina'),
        ('864632', 'Pomona'),
    ],
    # Csb（暖夏地中海气候）
    'Oregon_cities': [
        ('514547', 'Corvallis'),
        ('516721', 'Monmouth'),
        ('520047', 'Harrisburg'),
        ('523359', 'Albany'),
        ('523364', 'Tangent'),
        ('523372', 'Halsey'),
        ('528905', 'Jefferson'),
        ('534476', 'Lebanon'),
    ],
    # Bsk（冷半干旱气候）
    'Texas_cities' : [
        ('2267454', 'Dumas'),
        ('2271734', 'Canyon'),
        ('2282335', 'Amarillo'),
        ('2305727', 'Fritch'),
        ('2322802', 'Stinnett'),
        ('2329223', 'Borger'),
        ('2333532', 'Claude'),
        ('2374288', 'Pampa'),
    ]
}

# SELECTED_STATES = ['California_cities', 'Texas_cities', 'Oregon_cities']  # 可配置需要下载的州
SELECTED_STATES = ['Illinois_cities']  # 可配置需要下载的州
OUTPUT_ROOT = "./GHI_data"  # 根目录
TEMP_DIR = "./temp_downloads"


def main():
    Path(TEMP_DIR).mkdir(exist_ok=True)

    # 基础请求参数模板
    base_input_data = {
        'attributes': 'air_temperature,clearsky_ghi,cloud_type,dew_point,ghi,relative_humidity,solar_zenith_angle,total_precipitable_water,wind_direction,wind_speed',
        'interval': '10',
        'api_key': API_KEY,
        'email': EMAIL,
        'include_leap_day': 'true',
        'to_utc': 'false'
    }

    for state in SELECTED_STATES:
        state_sites = POINTS_BY_STATE.get(state, [])
        if not state_sites:
            print(f"[警告] 未找到 {state} 州的配置")
            continue

        # 创建州目录
        state_dir = Path(OUTPUT_ROOT) / state
        state_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== 开始处理 {state} 州数据 ===")

        # for year in ['2018', '2019', '2020', '2021', '2022']:
        for year in ['2019', '2020']:
            print(f"\n=== 处理 {year} 年数据 ===")
            for site_id, site_name in state_sites:
                try:
                    # 创建请求数据
                    input_data = copy.deepcopy(base_input_data)
                    input_data.update({'names': [year], 'location_ids': site_id})

                    headers = {'x-api-key': API_KEY}
                    response = requests.post(BASE_URL, data=input_data, headers=headers)
                    data = handle_api_response(response)
                    download_url = data['outputs']['downloadUrl']
                    print(f"[成功] 获取到 {state}-{site_name}({site_id}) {year} 下载链接")
                    process_compressed_data(download_url, site_id, year, state)
                except Exception as e:
                    print(f"[错误] {state}-{site_id} {year} 处理失败：{str(e)}")
                    continue
                time.sleep(1.5)


def process_compressed_data(url: str, site_id: str, year: str, state: str):
    """处理压缩文件并解压到对应州目录"""
    temp_file = download_temp_file(url, site_id, year)

    # 创建州/站点目录结构
    state_dir = Path(OUTPUT_ROOT) / state
    site_dir = state_dir / site_id
    site_dir.mkdir(exist_ok=True)

    if url.endswith('.zip'):
        extract_zip_flat(temp_file, site_dir, year)
    elif url.endswith('.gz'):
        extract_gz_flat(temp_file, site_dir, year)
    else:
        raise ValueError("不支持的压缩格式")


# 以下函数保持不变（与原始版本相同）
def download_temp_file(url: str, site_id: str, year: str) -> Path:
    temp_filename = f"{site_id}_{year}{os.path.splitext(url)[1]}"
    temp_path = Path(TEMP_DIR) / temp_filename

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(temp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return temp_path


def extract_zip_flat(zip_path: Path, target_dir: Path, year: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir() or os.path.basename(file_info.filename).startswith('.'):
                continue
            clean_name = os.path.basename(file_info.filename)
            year_prefix_name = f"{year}_{clean_name}"
            source = zip_ref.open(file_info)
            target = open(target_dir / year_prefix_name, "wb")
            with source, target:
                shutil.copyfileobj(source, target)
    print(f"ZIP文件已解压到：{target_dir}（共{len(zip_ref.namelist())}个文件）")


def extract_gz_flat(gz_path: Path, target_dir: Path, year: str):
    original_name = gz_path.stem
    output_name = f"{year}_{original_name}"
    output_path = target_dir / output_name

    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"GZ文件已解压到：{output_path}")


def handle_api_response(response: requests.Response) -> dict:
    if response.status_code != 200:
        raise Exception(f"API响应异常：{response.status_code} {response.reason}")
    try:
        data = response.json()
    except:
        raise Exception("响应非JSON格式")
    if data.get('errors'):
        errors = '\n'.join(data['errors'])
        raise Exception(f"API返回错误：{errors}")
    return data


def cleanup_temp_files():
    if Path(TEMP_DIR).exists():
        shutil.rmtree(TEMP_DIR)
        print("临时文件已清理")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_temp_files()
