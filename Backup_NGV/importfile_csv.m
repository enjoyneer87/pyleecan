function  TorquemapOPmap1=importfile_csv(filename, dataLines)
%IMPORTFILE 텍스트 파일에서 데이터 가져오기
%  TORQUEMAPOPMAP1 = IMPORTFILE(FILENAME)은 디폴트 선택 사항에 따라 텍스트 파일
%  FILENAME에서 데이터를 읽습니다.  데이터를 테이블로 반환합니다.
%
%  TORQUEMAPOPMAP1 = IMPORTFILE(FILE, DATALINES)는 텍스트 파일 FILENAME의 데이터를
%  지정된 행 간격으로 읽습니다. DATALINES를 양의 정수 스칼라로 지정하거나 양의 정수 스칼라로 구성된 N×2
%  배열(인접하지 않은 행 간격인 경우)로 지정하십시오.
%
%  예:
%  TorquemapOPmap1 = importfile("Z:\01_Codes_Projects\git_pyleecan\Torque_map_OP_map.csv", [1, Inf]);
%
%  READTABLE도 참조하십시오.
%
% MATLAB에서 2022-07-08 23:26:08에 자동 생성됨

%% 입력 처리

% dataLines를 지정하지 않는 경우 디폴트 값을 정의하십시오.
if nargin < 2
    dataLines = [1, Inf];
end

%% 가져오기 옵션을 설정하고 데이터 가져오기
opts = delimitedTextImportOptions("NumVariables", 4);

% 범위 및 구분 기호 지정
opts.DataLines = dataLines;
opts.Delimiter = ",";

% 열 이름과 유형 지정
opts.VariableNames = ["rpm", "id", "iq", "Torq"];
opts.VariableTypes = ["double", "double", "double", "double"];

% 파일 수준 속성 지정
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 데이터 가져오기
TorquemapOPmap1 = readtable(filename, opts);


return 