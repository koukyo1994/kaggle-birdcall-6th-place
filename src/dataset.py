import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data

from pathlib import Path


BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

NAME2CODE = {
    'Empidonax alnorum_Alder Flycatcher': 'aldfly',
    'Recurvirostra americana_American Avocet': 'ameavo',
    'Botaurus lentiginosus_American Bittern': 'amebit',
    'Corvus brachyrhynchos_American Crow': 'amecro',
    'Spinus tristis_American Goldfinch': 'amegfi',
    'Falco sparverius_American Kestrel': 'amekes',
    'Anthus rubescens_American Pipit': 'amepip',
    'Setophaga ruticilla_American Redstart': 'amered',
    'Turdus migratorius_American Robin': 'amerob',
    'Mareca americana_American Wigeon': 'amewig',
    'Scolopax minor_American Woodcock': 'amewoo',
    'Spizelloides arborea_American Tree Sparrow': 'amtspa',
    "Calypte anna_Anna's Hummingbird": 'annhum',
    'Myiarchus cinerascens_Ash-throated Flycatcher': 'astfly',
    "Calidris bairdii_Baird's Sandpiper": 'baisan',
    'Haliaeetus leucocephalus_Bald Eagle': 'baleag',
    'Icterus galbula_Baltimore Oriole': 'balori',
    'Riparia riparia_Bank Swallow': 'banswa',
    'Hirundo rustica_Barn Swallow': 'barswa',
    'Mniotilta varia_Black-and-white Warbler': 'bawwar',
    'Megaceryle alcyon_Belted Kingfisher': 'belkin1',
    "Artemisiospiza belli_Bell's Sparrow": 'belspa2',
    "Thryomanes bewickii_Bewick's Wren": 'bewwre',
    'Coccyzus erythropthalmus_Black-billed Cuckoo': 'bkbcuc',
    'Pica hudsonia_Black-billed Magpie': 'bkbmag1',
    'Setophaga fusca_Blackburnian Warbler': 'bkbwar',
    'Poecile atricapillus_Black-capped Chickadee': 'bkcchi',
    'Archilochus alexandri_Black-chinned Hummingbird': 'bkchum',
    'Pheucticus melanocephalus_Black-headed Grosbeak': 'bkhgro',
    'Setophaga striata_Blackpoll Warbler': 'bkpwar',
    'Amphispiza bilineata_Black-throated Sparrow': 'bktspa',
    'Sayornis nigricans_Black Phoebe': 'blkpho',
    'Passerina caerulea_Blue Grosbeak': 'blugrb1',
    'Cyanocitta cristata_Blue Jay': 'blujay',
    'Molothrus ater_Brown-headed Cowbird': 'bnhcow',
    'Dolichonyx oryzivorus_Bobolink': 'boboli',
    "Chroicocephalus philadelphia_Bonaparte's Gull": 'bongul',
    'Strix varia_Barred Owl': 'brdowl',
    "Euphagus cyanocephalus_Brewer's Blackbird": 'brebla',
    "Spizella breweri_Brewer's Sparrow": 'brespa',
    'Certhia americana_Brown Creeper': 'brncre',
    'Toxostoma rufum_Brown Thrasher': 'brnthr',
    'Selasphorus platycercus_Broad-tailed Hummingbird':
    'brthum', 'Buteo platypterus_Broad-winged Hawk': 'brwhaw',
    'Setophaga caerulescens_Black-throated Blue Warbler': 'btbwar',
    'Setophaga virens_Black-throated Green Warbler': 'btnwar',
    'Setophaga nigrescens_Black-throated Gray Warbler': 'btywar',
    'Bucephala albeola_Bufflehead': 'buffle',
    'Polioptila caerulea_Blue-gray Gnatcatcher': 'buggna',
    'Vireo solitarius_Blue-headed Vireo': 'buhvir',
    "Icterus bullockii_Bullock's Oriole": 'bulori',
    'Psaltriparus minimus_Bushtit': 'bushti',
    'Spatula discors_Blue-winged Teal': 'buwtea',
    'Vermivora cyanoptera_Blue-winged Warbler': 'buwwar',
    'Campylorhynchus brunneicapillus_Cactus Wren': 'cacwre',
    'Larus californicus_California Gull': 'calgul',
    'Callipepla californica_California Quail': 'calqua',
    'Setophaga tigrina_Cape May Warbler': 'camwar',
    'Branta canadensis_Canada Goose': 'cangoo',
    'Cardellina canadensis_Canada Warbler': 'canwar',
    'Catherpes mexicanus_Canyon Wren': 'canwre',
    'Thryothorus ludovicianus_Carolina Wren': 'carwre',
    "Haemorhous cassinii_Cassin's Finch": 'casfin',
    'Hydroprogne caspia_Caspian Tern': 'caster1',
    "Vireo cassinii_Cassin's Vireo": 'casvir',
    'Bombycilla cedrorum_Cedar Waxwing': 'cedwax',
    'Spizella passerina_Chipping Sparrow': 'chispa',
    'Chaetura pelagica_Chimney Swift': 'chiswi',
    'Setophaga pensylvanica_Chestnut-sided Warbler': 'chswar',
    'Alectoris chukar_Chukar': 'chukar',
    "Nucifraga columbiana_Clark's Nutcracker": 'clanut',
    'Petrochelidon pyrrhonota_Cliff Swallow': 'cliswa',
    'Bucephala clangula_Common Goldeneye': 'comgol',
    'Quiscalus quiscula_Common Grackle': 'comgra',
    'Gavia immer_Common Loon': 'comloo',
    'Mergus merganser_Common Merganser': 'commer',
    'Chordeiles minor_Common Nighthawk': 'comnig',
    'Corvus corax_Common Raven': 'comrav',
    'Acanthis flammea_Common Redpoll': 'comred',
    'Sterna hirundo_Common Tern': 'comter',
    'Geothlypis trichas_Common Yellowthroat': 'comyel',
    "Accipiter cooperii_Cooper's Hawk": 'coohaw',
    "Calypte costae_Costa's Hummingbird": 'coshum',
    'Aphelocoma californica_California Scrub-Jay': 'cowscj1',
    'Junco hyemalis_Dark-eyed Junco': 'daejun',
    'Phalacrocorax auritus_Double-crested Cormorant': 'doccor',
    'Dryobates pubescens_Downy Woodpecker': 'dowwoo',
    'Empidonax oberholseri_Dusky Flycatcher': 'dusfly',
    'Podiceps nigricollis_Eared Grebe': 'eargre',
    'Sialia sialis_Eastern Bluebird': 'easblu',
    'Tyrannus tyrannus_Eastern Kingbird': 'easkin',
    'Sturnella magna_Eastern Meadowlark': 'easmea',
    'Sayornis phoebe_Eastern Phoebe': 'easpho',
    'Pipilo erythrophthalmus_Eastern Towhee': 'eastow',
    'Contopus virens_Eastern Wood-Pewee': 'eawpew',
    'Streptopelia decaocto_Eurasian Collared-Dove': 'eucdov',
    'Sturnus vulgaris_European Starling': 'eursta',
    'Coccothraustes vespertinus_Evening Grosbeak': 'evegro',
    'Spizella pusilla_Field Sparrow': 'fiespa',
    'Corvus ossifragus_Fish Crow': 'fiscro',
    'Passerella iliaca_Fox Sparrow': 'foxspa',
    'Mareca strepera_Gadwall': 'gadwal',
    'Leucosticte tephrocotis_Gray-crowned Rosy-Finch': 'gcrfin',
    'Pipilo chlorurus_Green-tailed Towhee': 'gnttow',
    'Anas crecca_Green-winged Teal': 'gnwtea',
    'Regulus satrapa_Golden-crowned Kinglet': 'gockin',
    'Zonotrichia atricapilla_Golden-crowned Sparrow': 'gocspa',
    'Aquila chrysaetos_Golden Eagle': 'goleag',
    'Ardea herodias_Great Blue Heron': 'grbher3',
    'Myiarchus crinitus_Great Crested Flycatcher': 'grcfly',
    'Ardea alba_Great Egret': 'greegr',
    'Geococcyx californianus_Greater Roadrunner': 'greroa',
    'Tringa melanoleuca_Greater Yellowlegs': 'greyel',
    'Bubo virginianus_Great Horned Owl': 'grhowl',
    'Butorides virescens_Green Heron': 'grnher',
    'Quiscalus mexicanus_Great-tailed Grackle': 'grtgra',
    'Dumetella carolinensis_Gray Catbird': 'grycat',
    'Empidonax wrightii_Gray Flycatcher': 'gryfly',
    'Dryobates villosus_Hairy Woodpecker': 'haiwoo',
    "Empidonax hammondii_Hammond's Flycatcher": 'hamfly',
    'Larus argentatus_Herring Gull': 'hergul',
    'Catharus guttatus_Hermit Thrush': 'herthr',
    'Lophodytes cucullatus_Hooded Merganser': 'hoomer',
    'Setophaga citrina_Hooded Warbler': 'hoowar',
    'Podiceps auritus_Horned Grebe': 'horgre',
    'Eremophila alpestris_Horned Lark': 'horlar',
    'Haemorhous mexicanus_House Finch': 'houfin',
    'Passer domesticus_House Sparrow': 'houspa',
    'Troglodytes aedon_House Wren': 'houwre',
    'Passerina cyanea_Indigo Bunting': 'indbun',
    'Baeolophus ridgwayi_Juniper Titmouse': 'juntit1',
    'Charadrius vociferus_Killdeer': 'killde',
    'Dryobates scalaris_Ladder-backed Woodpecker': 'labwoo',
    'Chondestes grammacus_Lark Sparrow': 'larspa',
    'Passerina amoena_Lazuli Bunting': 'lazbun',
    'Ixobrychus exilis_Least Bittern': 'leabit',
    'Empidonax minimus_Least Flycatcher': 'leafly',
    'Calidris minutilla_Least Sandpiper': 'leasan',
    "Toxostoma lecontei_LeConte's Thrasher": 'lecthr',
    'Spinus psaltria_Lesser Goldfinch': 'lesgol',
    'Chordeiles acutipennis_Lesser Nighthawk': 'lesnig',
    'Tringa flavipes_Lesser Yellowlegs': 'lesyel',
    "Melanerpes lewis_Lewis's Woodpecker": 'lewwoo',
    "Melospiza lincolnii_Lincoln's Sparrow": 'linspa',
    'Numenius americanus_Long-billed Curlew': 'lobcur',
    'Limnodromus scolopaceus_Long-billed Dowitcher': 'lobdow',
    'Lanius ludovicianus_Loggerhead Shrike': 'logshr',
    'Clangula hyemalis_Long-tailed Duck': 'lotduc',
    'Parkesia motacilla_Louisiana Waterthrush': 'louwat',
    "Geothlypis tolmiei_MacGillivray's Warbler": 'macwar',
    'Setophaga magnolia_Magnolia Warbler': 'magwar',
    'Anas platyrhynchos_Mallard': 'mallar3',
    'Cistothorus palustris_Marsh Wren': 'marwre',
    'Falco columbarius_Merlin': 'merlin',
    'Sialia currucoides_Mountain Bluebird': 'moublu',
    'Poecile gambeli_Mountain Chickadee': 'mouchi',
    'Zenaida macroura_Mourning Dove': 'moudov',
    'Cardinalis cardinalis_Northern Cardinal': 'norcar',
    'Colaptes auratus_Northern Flicker': 'norfli',
    'Circus hudsonius_Northern Harrier': 'norhar2',
    'Mimus polyglottos_Northern Mockingbird': 'normoc',
    'Setophaga americana_Northern Parula': 'norpar',
    'Anas acuta_Northern Pintail': 'norpin',
    'Spatula clypeata_Northern Shoveler': 'norsho',
    'Parkesia noveboracensis_Northern Waterthrush': 'norwat',
    'Stelgidopteryx serripennis_Northern Rough-winged Swallow': 'nrwswa',
    "Dryobates nuttallii_Nuttall's Woodpecker": 'nutwoo',
    'Contopus cooperi_Olive-sided Flycatcher': 'olsfly',
    'Leiothlypis celata_Orange-crowned Warbler': 'orcwar',
    'Pandion haliaetus_Osprey': 'osprey',
    'Seiurus aurocapilla_Ovenbird': 'ovenbi1',
    'Setophaga palmarum_Palm Warbler': 'palwar',
    'Empidonax difficilis_Pacific-slope Flycatcher': 'pasfly',
    'Calidris melanotos_Pectoral Sandpiper': 'pecsan',
    'Falco peregrinus_Peregrine Falcon': 'perfal',
    'Phainopepla nitens_Phainopepla': 'phaino',
    'Podilymbus podiceps_Pied-billed Grebe': 'pibgre',
    'Dryocopus pileatus_Pileated Woodpecker': 'pilwoo',
    'Pinicola enucleator_Pine Grosbeak': 'pingro',
    'Gymnorhinus cyanocephalus_Pinyon Jay': 'pinjay',
    'Spinus pinus_Pine Siskin': 'pinsis',
    'Setophaga pinus_Pine Warbler': 'pinwar',
    'Vireo plumbeus_Plumbeous Vireo': 'plsvir',
    'Setophaga discolor_Prairie Warbler': 'prawar',
    'Haemorhous purpureus_Purple Finch': 'purfin',
    'Sitta pygmaea_Pygmy Nuthatch': 'pygnut',
    'Mergus serrator_Red-breasted Merganser': 'rebmer',
    'Sitta canadensis_Red-breasted Nuthatch': 'rebnut',
    'Sphyrapicus ruber_Red-breasted Sapsucker': 'rebsap',
    'Melanerpes carolinus_Red-bellied Woodpecker': 'rebwoo',
    'Loxia curvirostra_Red Crossbill': 'redcro',
    'Aythya americana_Redhead': 'redhea',
    'Vireo olivaceus_Red-eyed Vireo': 'reevir1',
    'Phalaropus lobatus_Red-necked Phalarope': 'renpha',
    'Buteo lineatus_Red-shouldered Hawk': 'reshaw',
    'Buteo jamaicensis_Red-tailed Hawk': 'rethaw',
    'Agelaius phoeniceus_Red-winged Blackbird': 'rewbla',
    'Larus delawarensis_Ring-billed Gull': 'ribgul',
    'Aythya collaris_Ring-necked Duck': 'rinduc',
    'Pheucticus ludovicianus_Rose-breasted Grosbeak': 'robgro',
    'Columba livia_Rock Pigeon': 'rocpig',
    'Salpinctes obsoletus_Rock Wren': 'rocwre',
    'Archilochus colubris_Ruby-throated Hummingbird': 'rthhum',
    'Regulus calendula_Ruby-crowned Kinglet': 'ruckin',
    'Oxyura jamaicensis_Ruddy Duck': 'rudduc',
    'Bonasa umbellus_Ruffed Grouse': 'rufgro',
    'Selasphorus rufus_Rufous Hummingbird': 'rufhum',
    'Euphagus carolinus_Rusty Blackbird': 'rusbla',
    'Artemisiospiza nevadensis_Sagebrush Sparrow': 'sagspa1',
    'Oreoscoptes montanus_Sage Thrasher': 'sagthr',
    'Passerculus sandwichensis_Savannah Sparrow': 'savspa',
    "Sayornis saya_Say's Phoebe": 'saypho',
    'Piranga olivacea_Scarlet Tanager': 'scatan',
    "Icterus parisorum_Scott's Oriole": 'scoori',
    'Charadrius semipalmatus_Semipalmated Plover': 'semplo',
    'Calidris pusilla_Semipalmated Sandpiper': 'semsan',
    'Asio flammeus_Short-eared Owl': 'sheowl',
    'Accipiter striatus_Sharp-shinned Hawk': 'shshaw',
    'Plectrophenax nivalis_Snow Bunting': 'snobun',
    'Anser caerulescens_Snow Goose': 'snogoo',
    'Tringa solitaria_Solitary Sandpiper': 'solsan',
    'Melospiza melodia_Song Sparrow': 'sonspa',
    'Porzana carolina_Sora': 'sora',
    'Actitis macularius_Spotted Sandpiper': 'sposan',
    'Pipilo maculatus_Spotted Towhee': 'spotow',
    "Cyanocitta stelleri_Steller's Jay": 'stejay',
    "Buteo swainsoni_Swainson's Hawk": 'swahaw',
    'Melospiza georgiana_Swamp Sparrow': 'swaspa',
    "Catharus ustulatus_Swainson's Thrush": 'swathr',
    'Tachycineta bicolor_Tree Swallow': 'treswa',
    'Cygnus buccinator_Trumpeter Swan': 'truswa',
    'Baeolophus bicolor_Tufted Titmouse': 'tuftit',
    'Cygnus columbianus_Tundra Swan': 'tunswa',
    'Catharus fuscescens_Veery': 'veery',
    'Pooecetes gramineus_Vesper Sparrow': 'vesspa',
    'Tachycineta thalassina_Violet-green Swallow': 'vigswa',
    'Vireo gilvus_Warbling Vireo': 'warvir',
    'Sialia mexicana_Western Bluebird': 'wesblu',
    'Aechmophorus occidentalis_Western Grebe': 'wesgre',
    'Tyrannus verticalis_Western Kingbird': 'weskin',
    'Sturnella neglecta_Western Meadowlark': 'wesmea',
    'Calidris mauri_Western Sandpiper': 'wessan',
    'Piranga ludoviciana_Western Tanager': 'westan',
    'Contopus sordidulus_Western Wood-Pewee': 'wewpew',
    'Sitta carolinensis_White-breasted Nuthatch': 'whbnut',
    'Zonotrichia leucophrys_White-crowned Sparrow': 'whcspa',
    'Plegadis chihi_White-faced Ibis': 'whfibi',
    'Zonotrichia albicollis_White-throated Sparrow': 'whtspa',
    'Aeronautes saxatalis_White-throated Swift': 'whtswi',
    'Empidonax traillii_Willow Flycatcher': 'wilfly',
    "Gallinago delicata_Wilson's Snipe": 'wilsni1',
    'Meleagris gallopavo_Wild Turkey': 'wiltur',
    'Troglodytes hiemalis_Winter Wren': 'winwre3',
    "Cardellina pusilla_Wilson's Warbler": 'wlswar',
    'Aix sponsa_Wood Duck': 'wooduc',
    "Aphelocoma woodhouseii_Woodhouse's Scrub-Jay": 'wooscj2',
    'Hylocichla mustelina_Wood Thrush': 'woothr',
    'Fulica americana_American Coot': 'y00475',
    'Empidonax flaviventris_Yellow-bellied Flycatcher': 'yebfly',
    'Sphyrapicus varius_Yellow-bellied Sapsucker': 'yebsap',
    'Xanthocephalus xanthocephalus_Yellow-headed Blackbird': 'yehbla',
    'Setophaga petechia_Yellow Warbler': 'yelwar',
    'Setophaga coronata_Yellow-rumped Warbler': 'yerwar',
    'Vireo flavifrons_Yellow-throated Vireo': 'yetvir'
}

SCINAME2CODE = {
    'Empidonax alnorum': 'aldfly', 'Recurvirostra americana': 'ameavo',
    'Botaurus lentiginosus': 'amebit', 'Corvus brachyrhynchos': 'amecro',
    'Spinus tristis': 'amegfi', 'Falco sparverius': 'amekes',
    'Anthus rubescens': 'amepip', 'Setophaga ruticilla': 'amered',
    'Turdus migratorius': 'amerob', 'Mareca americana': 'amewig',
    'Scolopax minor': 'amewoo', 'Spizelloides arborea': 'amtspa',
    'Calypte anna': 'annhum', 'Myiarchus cinerascens': 'astfly',
    'Calidris bairdii': 'baisan', 'Haliaeetus leucocephalus': 'baleag',
    'Icterus galbula': 'balori', 'Riparia riparia': 'banswa',
    'Hirundo rustica': 'barswa', 'Mniotilta varia': 'bawwar',
    'Megaceryle alcyon': 'belkin1', 'Artemisiospiza belli': 'belspa2',
    'Thryomanes bewickii': 'bewwre', 'Coccyzus erythropthalmus': 'bkbcuc',
    'Pica hudsonia': 'bkbmag1', 'Setophaga fusca': 'bkbwar',
    'Poecile atricapillus': 'bkcchi', 'Archilochus alexandri': 'bkchum',
    'Pheucticus melanocephalus': 'bkhgro', 'Setophaga striata': 'bkpwar',
    'Amphispiza bilineata': 'bktspa', 'Sayornis nigricans': 'blkpho',
    'Passerina caerulea': 'blugrb1', 'Cyanocitta cristata': 'blujay',
    'Molothrus ater': 'bnhcow', 'Dolichonyx oryzivorus': 'boboli',
    'Chroicocephalus philadelphia': 'bongul', 'Strix varia': 'brdowl',
    'Euphagus cyanocephalus': 'brebla', 'Spizella breweri': 'brespa',
    'Certhia americana': 'brncre', 'Toxostoma rufum': 'brnthr',
    'Selasphorus platycercus': 'brthum', 'Buteo platypterus': 'brwhaw',
    'Setophaga caerulescens': 'btbwar', 'Setophaga virens': 'btnwar',
    'Setophaga nigrescens': 'btywar', 'Bucephala albeola': 'buffle',
    'Polioptila caerulea': 'buggna', 'Vireo solitarius': 'buhvir',
    'Icterus bullockii': 'bulori', 'Psaltriparus minimus': 'bushti',
    'Spatula discors': 'buwtea', 'Vermivora cyanoptera': 'buwwar',
    'Campylorhynchus brunneicapillus': 'cacwre', 'Larus californicus': 'calgul',
    'Callipepla californica': 'calqua', 'Setophaga tigrina': 'camwar',
    'Branta canadensis': 'cangoo', 'Cardellina canadensis': 'canwar',
    'Catherpes mexicanus': 'canwre', 'Thryothorus ludovicianus': 'carwre',
    'Haemorhous cassinii': 'casfin', 'Hydroprogne caspia': 'caster1',
    'Vireo cassinii': 'casvir', 'Bombycilla cedrorum': 'cedwax',
    'Spizella passerina': 'chispa', 'Chaetura pelagica': 'chiswi',
    'Setophaga pensylvanica': 'chswar', 'Alectoris chukar': 'chukar',
    'Nucifraga columbiana': 'clanut', 'Petrochelidon pyrrhonota': 'cliswa',
    'Bucephala clangula': 'comgol', 'Quiscalus quiscula': 'comgra',
    'Gavia immer': 'comloo', 'Mergus merganser': 'commer',
    'Chordeiles minor': 'comnig', 'Corvus corax': 'comrav',
    'Acanthis flammea': 'comred', 'Sterna hirundo': 'comter',
    'Geothlypis trichas': 'comyel', 'Accipiter cooperii': 'coohaw',
    'Calypte costae': 'coshum', 'Aphelocoma californica': 'cowscj1',
    'Junco hyemalis': 'daejun', 'Phalacrocorax auritus': 'doccor',
    'Dryobates pubescens': 'dowwoo', 'Empidonax oberholseri': 'dusfly',
    'Podiceps nigricollis': 'eargre', 'Sialia sialis': 'easblu',
    'Tyrannus tyrannus': 'easkin', 'Sturnella magna': 'easmea',
    'Sayornis phoebe': 'easpho', 'Pipilo erythrophthalmus': 'eastow',
    'Contopus virens': 'eawpew', 'Streptopelia decaocto': 'eucdov',
    'Sturnus vulgaris': 'eursta', 'Hesperiphona vespertina': 'evegro',
    'Spizella pusilla': 'fiespa', 'Corvus ossifragus': 'fiscro',
    'Passerella iliaca': 'foxspa', 'Mareca strepera': 'gadwal',
    'Leucosticte tephrocotis': 'gcrfin', 'Pipilo chlorurus': 'gnttow',
    'Anas crecca': 'gnwtea', 'Regulus satrapa': 'gockin',
    'Zonotrichia atricapilla': 'gocspa', 'Aquila chrysaetos': 'goleag',
    'Ardea herodias': 'grbher3', 'Myiarchus crinitus': 'grcfly',
    'Ardea alba': 'greegr', 'Geococcyx californianus': 'greroa',
    'Tringa melanoleuca': 'greyel', 'Bubo virginianus': 'grhowl',
    'Butorides virescens': 'grnher', 'Quiscalus mexicanus': 'grtgra',
    'Dumetella carolinensis': 'grycat', 'Empidonax wrightii': 'gryfly',
    'Leuconotopicus villosus': 'haiwoo', 'Empidonax hammondii': 'hamfly',
    'Larus argentatus': 'hergul', 'Catharus guttatus': 'herthr',
    'Lophodytes cucullatus': 'hoomer', 'Setophaga citrina': 'hoowar',
    'Podiceps auritus': 'horgre', 'Eremophila alpestris': 'horlar',
    'Haemorhous mexicanus': 'houfin', 'Passer domesticus': 'houspa',
    'Troglodytes aedon': 'houwre', 'Passerina cyanea': 'indbun',
    'Baeolophus ridgwayi': 'juntit1', 'Charadrius vociferus': 'killde',
    'Dryobates scalaris': 'labwoo', 'Chondestes grammacus': 'larspa',
    'Passerina amoena': 'lazbun', 'Ixobrychus exilis': 'leabit',
    'Empidonax minimus': 'leafly', 'Calidris minutilla': 'leasan',
    'Toxostoma lecontei': 'lecthr', 'Spinus psaltria': 'lesgol',
    'Chordeiles acutipennis': 'lesnig', 'Tringa flavipes': 'lesyel',
    'Melanerpes lewis': 'lewwoo', 'Melospiza lincolnii': 'linspa',
    'Numenius americanus': 'lobcur', 'Limnodromus scolopaceus': 'lobdow',
    'Lanius ludovicianus': 'logshr', 'Clangula hyemalis': 'lotduc',
    'Parkesia motacilla': 'louwat', 'Geothlypis tolmiei': 'macwar',
    'Setophaga magnolia': 'magwar', 'Anas platyrhynchos': 'mallar3',
    'Cistothorus palustris': 'marwre', 'Falco columbarius': 'merlin',
    'Sialia currucoides': 'moublu', 'Poecile gambeli': 'mouchi',
    'Zenaida macroura': 'moudov', 'Cardinalis cardinalis': 'norcar',
    'Colaptes auratus': 'norfli', 'Circus hudsonius': 'norhar2',
    'Mimus polyglottos': 'normoc', 'Setophaga americana': 'norpar',
    'Anas acuta': 'norpin', 'Spatula clypeata': 'norsho',
    'Parkesia noveboracensis': 'norwat', 'Stelgidopteryx serripennis': 'nrwswa',
    'Dryobates nuttallii': 'nutwoo', 'Contopus cooperi': 'olsfly',
    'Leiothlypis celata': 'orcwar', 'Pandion haliaetus': 'osprey',
    'Seiurus aurocapilla': 'ovenbi1', 'Setophaga palmarum': 'palwar',
    'Empidonax difficilis': 'pasfly', 'Calidris melanotos': 'pecsan',
    'Falco peregrinus': 'perfal', 'Phainopepla nitens': 'phaino',
    'Podilymbus podiceps': 'pibgre', 'Dryocopus pileatus': 'pilwoo',
    'Pinicola enucleator': 'pingro', 'Gymnorhinus cyanocephalus': 'pinjay',
    'Spinus pinus': 'pinsis', 'Setophaga pinus': 'pinwar',
    'Vireo plumbeus': 'plsvir', 'Setophaga discolor': 'prawar',
    'Haemorhous purpureus': 'purfin', 'Sitta pygmaea': 'pygnut',
    'Mergus serrator': 'rebmer', 'Sitta canadensis': 'rebnut',
    'Sphyrapicus ruber': 'rebsap', 'Melanerpes carolinus': 'rebwoo',
    'Loxia curvirostra': 'redcro', 'Aythya americana': 'redhea',
    'Vireo olivaceus': 'reevir1', 'Phalaropus lobatus': 'renpha',
    'Buteo lineatus': 'reshaw', 'Buteo jamaicensis': 'rethaw',
    'Agelaius phoeniceus': 'rewbla', 'Larus delawarensis': 'ribgul',
    'Aythya collaris': 'rinduc', 'Pheucticus ludovicianus': 'robgro',
    'Columba livia': 'rocpig', 'Salpinctes obsoletus': 'rocwre',
    'Archilochus colubris': 'rthhum', 'Regulus calendula': 'ruckin',
    'Oxyura jamaicensis': 'rudduc', 'Bonasa umbellus': 'rufgro',
    'Selasphorus rufus': 'rufhum', 'Euphagus carolinus': 'rusbla',
    'Artemisiospiza nevadensis': 'sagspa1', 'Oreoscoptes montanus': 'sagthr',
    'Passerculus sandwichensis': 'savspa', 'Sayornis saya': 'saypho',
    'Piranga olivacea': 'scatan', 'Icterus parisorum': 'scoori',
    'Charadrius semipalmatus': 'semplo', 'Calidris pusilla': 'semsan',
    'Asio flammeus': 'sheowl', 'Accipiter striatus': 'shshaw',
    'Plectrophenax nivalis': 'snobun', 'Anser caerulescens': 'snogoo',
    'Tringa solitaria': 'solsan', 'Melospiza melodia': 'sonspa',
    'Porzana carolina': 'sora', 'Actitis macularius': 'sposan',
    'Pipilo maculatus': 'spotow', 'Cyanocitta stelleri': 'stejay',
    'Buteo swainsoni': 'swahaw', 'Melospiza georgiana': 'swaspa',
    'Catharus ustulatus': 'swathr', 'Tachycineta bicolor': 'treswa',
    'Cygnus buccinator': 'truswa', 'Baeolophus bicolor': 'tuftit',
    'Cygnus columbianus': 'tunswa', 'Catharus fuscescens': 'veery',
    'Pooecetes gramineus': 'vesspa', 'Tachycineta thalassina': 'vigswa',
    'Vireo gilvus': 'warvir', 'Sialia mexicana': 'wesblu',
    'Aechmophorus occidentalis': 'wesgre', 'Tyrannus verticalis': 'weskin',
    'Sturnella neglecta': 'wesmea', 'Calidris mauri': 'wessan',
    'Piranga ludoviciana': 'westan', 'Contopus sordidulus': 'wewpew',
    'Sitta carolinensis': 'whbnut', 'Zonotrichia leucophrys': 'whcspa',
    'Plegadis chihi': 'whfibi', 'Zonotrichia albicollis': 'whtspa',
    'Aeronautes saxatalis': 'whtswi', 'Empidonax traillii': 'wilfly',
    'Gallinago delicata': 'wilsni1', 'Meleagris gallopavo': 'wiltur',
    'Troglodytes hiemalis': 'winwre3', 'Cardellina pusilla': 'wlswar',
    'Aix sponsa': 'wooduc', 'Aphelocoma woodhouseii': 'wooscj2',
    'Hylocichla mustelina': 'woothr', 'Fulica americana': 'y00475',
    'Empidonax flaviventris': 'yebfly', 'Sphyrapicus varius': 'yebsap',
    'Xanthocephalus xanthocephalus': 'yehbla', 'Setophaga petechia': 'yelwar',
    'Setophaga coronata': 'yerwar', 'Vireo flavifrons': 'yetvir'
}

PERIOD = 5


class PANNsMultiLabelDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None, period=30):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])
        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.transforms:
            y = self.transforms(y)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for second_label in secondary_labels:
            if NAME2CODE.get(second_label) is not None:
                second_code = NAME2CODE[second_label]
                labels[BIRD_CODE[second_code]] = 1

        return {
            "waveform": y,
            "targets": labels
        }


class PANNsSedDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None,
                 denoised_audio_dir=None):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.denoised_audio_dir = denoised_audio_dir
        if denoised_audio_dir is not None:
            self.use_denoised = True
        else:
            self.use_denoised = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        if self.use_denoised:
            path = self.denoised_audio_dir / ebird_code / wav_name
            if path.exists():
                y, sr = sf.read(path)
            else:
                y, sr = sf.read(self.datadir / ebird_code / wav_name)
        else:
            y, sr = sf.read(self.datadir / ebird_code / wav_name)

        duration = len(y) / sr
        if self.transforms:
            y = self.transforms(y)

        audios = []
        len_y = len(y)
        start = 0
        end = sr * 5
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (sr * 5):
                y_batch_large = np.zeros(sr * 5, dtype=y_batch.dtype)
                y_batch_large[:len(y_batch)] = y_batch
                audios.append(y_batch_large)
                break
            start = end
            end = end + sr * 5

            audios.append(y_batch)
        audios = np.asarray(audios).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for secondary_label in secondary_labels:
            code = NAME2CODE.get(secondary_label)
            if code is None:
                continue
            else:
                labels[
                    BIRD_CODE[code]
                ] = 1

        return {
            "waveform": audios,
            "targets": labels,
            "ebird_code": ebird_code,
            "wav_name": wav_name,
            "duration": duration
        }


class NormalizedChannelsSedDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None,
                 denoised_audio_dir=None, melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.denoised_audio_dir = denoised_audio_dir
        if denoised_audio_dir is not None:
            self.use_denoised = True
        else:
            self.use_denoised = False
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        if self.use_denoised:
            path = self.denoised_audio_dir / ebird_code / wav_name
            if path.exists():
                y, sr = sf.read(path)
            else:
                y, sr = sf.read(self.datadir / ebird_code / wav_name)
        else:
            y, sr = sf.read(self.datadir / ebird_code / wav_name)

        duration = len(y) / sr
        if self.transforms:
            y = self.transforms(y)

        images = []
        len_y = len(y)
        if len(y) > 0:
            max_vol = np.abs(y).max()
            if max_vol > 0:
                y = np.asfortranarray(y * 1 / max_vol)
        start = 0
        end = sr * self.period
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (sr * self.period):
                y_batch_large = np.zeros(sr * self.period, dtype=y_batch.dtype)
                y_batch_large[:len(y_batch)] = y_batch

                melspec = librosa.feature.melspectrogram(
                    y_batch_large, sr=sr, **self.melspectrogram_parameters)
                pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
                clean_mel = librosa.power_to_db(melspec ** 1.5)
                melspec = librosa.power_to_db(melspec)

                norm_melspec = normalize_melspec(melspec)
                norm_pcen = normalize_melspec(pcen)
                norm_clean_mel = normalize_melspec(clean_mel)
                image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

                height, width, _ = image.shape
                image = cv2.resize(image, (int(width * 224 / height), 224))
                image = np.moveaxis(image, 2, 0)
                image = (image / 255.0).astype(np.float32)

                images.append(image)
                break
            start = end
            end = end + sr * self.period

            melspec = librosa.feature.melspectrogram(
                y_batch, sr=sr, **self.melspectrogram_parameters)
            pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
            clean_mel = librosa.power_to_db(melspec ** 1.5)
            melspec = librosa.power_to_db(melspec)

            norm_melspec = normalize_melspec(melspec)
            norm_pcen = normalize_melspec(pcen)
            norm_clean_mel = normalize_melspec(clean_mel)
            image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * 224 / height), 224))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)

            images.append(image)
        images = np.asarray(images).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for secondary_label in secondary_labels:
            code = NAME2CODE.get(secondary_label)
            if code is None:
                continue
            else:
                labels[
                    BIRD_CODE[code]
                ] = 1

        return {
            "image": images,
            "targets": labels,
            "ebird_code": ebird_code,
            "wav_name": wav_name,
            "duration": duration,
            "period": self.period
        }


class ChannelsSedDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None,
                 denoised_audio_dir=None, melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.denoised_audio_dir = denoised_audio_dir
        if denoised_audio_dir is not None:
            self.use_denoised = True
        else:
            self.use_denoised = False
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        if self.use_denoised:
            path = self.denoised_audio_dir / ebird_code / wav_name
            if path.exists():
                y, sr = sf.read(path)
            else:
                y, sr = sf.read(self.datadir / ebird_code / wav_name)
        else:
            y, sr = sf.read(self.datadir / ebird_code / wav_name)

        duration = len(y) / sr
        if self.transforms:
            y = self.transforms(y)

        images = []
        len_y = len(y)
        start = 0
        end = sr * self.period
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (sr * self.period):
                y_batch_large = np.zeros(sr * self.period, dtype=y_batch.dtype)
                y_batch_large[:len(y_batch)] = y_batch

                melspec = librosa.feature.melspectrogram(
                    y_batch_large, sr=sr, **self.melspectrogram_parameters)
                pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
                clean_mel = librosa.power_to_db(melspec ** 1.5)
                melspec = librosa.power_to_db(melspec)

                norm_melspec = normalize_melspec(melspec)
                norm_pcen = normalize_melspec(pcen)
                norm_clean_mel = normalize_melspec(clean_mel)
                image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

                height, width, _ = image.shape
                image = cv2.resize(image, (int(width * 224 / height), 224))
                image = np.moveaxis(image, 2, 0)
                image = (image / 255.0).astype(np.float32)

                images.append(image)
                break
            start = end
            end = end + sr * self.period

            melspec = librosa.feature.melspectrogram(
                y_batch, sr=sr, **self.melspectrogram_parameters)
            pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
            clean_mel = librosa.power_to_db(melspec ** 1.5)
            melspec = librosa.power_to_db(melspec)

            norm_melspec = normalize_melspec(melspec)
            norm_pcen = normalize_melspec(pcen)
            norm_clean_mel = normalize_melspec(clean_mel)
            image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * 224 / height), 224))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)

            images.append(image)
        images = np.asarray(images).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for secondary_label in secondary_labels:
            code = NAME2CODE.get(secondary_label)
            if code is None:
                continue
            else:
                labels[
                    BIRD_CODE[code]
                ] = 1

        return {
            "image": images,
            "targets": labels,
            "ebird_code": ebird_code,
            "wav_name": wav_name,
            "duration": duration,
            "period": self.period
        }


class MultiChannelDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
        clean_mel = librosa.power_to_db(melspec ** 1.5)
        melspec = librosa.power_to_db(melspec)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(image=melspec)["image"]
            pcen = self.spectrogram_transforms(image=pcen)["image"]
            clean_mel = self.spectrogram_transforms(image=clean_mel)["image"]
        else:
            pass

        norm_melspec = normalize_melspec(melspec)
        norm_pcen = normalize_melspec(pcen)
        norm_clean_mel = normalize_melspec(clean_mel)
        image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for second_label in secondary_labels:
            if NAME2CODE.get(second_label) is not None:
                second_code = NAME2CODE[second_label]
                labels[BIRD_CODE[second_code]] = 1

        return {
            "image": image,
            "targets": labels
        }


class LabelCorrectionDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 soft_label_dir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30,
                 n_segments=103,
                 threshold=0.5):
        self.df = df
        self.datadir = datadir
        self.soft_label_dir = soft_label_dir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period
        self.n_segments = n_segments
        self.threshold = threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        y, sr = sf.read(self.datadir / ebird_code / wav_name)
        soft_label = np.load(self.soft_label_dir / (wav_name + ".npy"))

        sec_per_segment = self.period / self.n_segments
        sec_per_timestep = 1 / sr
        step_per_segment = int(sec_per_segment / sec_per_timestep)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            max_offset = effective_length - len_y

            offset_id = np.random.randint(0, (max_offset // step_per_segment) + 1)
            start = offset_id * step_per_segment

            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            max_offset = len_y - effective_length

            offset_id = np.random.randint(0, (max_offset // step_per_segment) + 1)
            start = offset_id * step_per_segment
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
        clean_mel = librosa.power_to_db(melspec ** 1.5)
        melspec = librosa.power_to_db(melspec)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(image=melspec)["image"]
            pcen = self.spectrogram_transforms(image=pcen)["image"]
            clean_mel = self.spectrogram_transforms(image=clean_mel)["image"]
        else:
            pass

        norm_melspec = normalize_melspec(melspec)
        norm_pcen = normalize_melspec(pcen)
        norm_clean_mel = normalize_melspec(clean_mel)
        image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros([self.n_segments, len(BIRD_CODE)], dtype=np.float32)

        if len_y < effective_length:
            if len(soft_label) + offset_id >= len(labels):
                n_seg = len(labels[offset_id:, :])
                labels[offset_id:, :] = soft_label[:n_seg]
            else:
                labels[offset_id:offset_id + len(soft_label), :] = soft_label
        elif len_y > effective_length:
            use_labels = soft_label[offset_id:offset_id + len(labels)]
            if len(use_labels) < len(labels):
                labels[:len(use_labels)] = use_labels
            else:
                labels = use_labels
        else:
            if len(labels) >= len(soft_label):
                labels[:len(soft_label)] = soft_label
            else:
                labels = soft_label[:len(labels)]

        labels = labels.astype(np.float32)

        weak_labels = np.zeros(len(BIRD_CODE), dtype=int)
        weak_labels[BIRD_CODE[ebird_code]] = 1
        for second_label in secondary_labels:
            if NAME2CODE.get(second_label) is not None:
                second_code = NAME2CODE[second_label]
                weak_labels[BIRD_CODE[second_code]] = 1
        weak_labels_soft = labels.max(axis=0)
        weak_labels_bin = (weak_labels_soft >= self.threshold).astype(int)

        weak_labels = np.logical_and(weak_labels, weak_labels_bin).astype(int)
        weak_sum_target = labels.sum(axis=0)

        return {
            "image": image,
            "targets": weak_labels,
            "weak_targets": weak_labels,
            "weak_sum_targets": weak_sum_target
        }


def normalize_melspec(X: np.ndarray):
    eps = 1e-6
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    norm_min, norm_max = Xstd.min(), Xstd.max()
    if (norm_max - norm_min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
