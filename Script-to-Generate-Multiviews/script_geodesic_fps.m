%Add libraries
cd 'C:\Users\jefri\Documents\toolbox_graph\toolbox_fast_marching';
path(path,'C:\Users\jefri\Documents\toolbox_graph\toolbox_signal');
path(path,'C:\Users\jefri\Documents\toolbox_graph\toolbox_graph');
path(path,'C:\Users\jefri\Documents\toolbox_graph\toolbox_fast_marching');
path(path, 'toolbox/');

%SOURCE_DATASET = 'D:\Peruvian_Dataset\';
SOURCE_DATASET = 'D:\Pottery_Dataset\';
VERTEX_ORIGIN_CRITERION  = 'max'; % or max
m = 512;% the limit of FPS points
FPS_POINTS = [1 2 32 64 128 256 m];% in what number of FPS get the images
%m = 1; FPS_POINTS = [1];
mkdir 'D:\Pottery_Result';%output folder
groups = {'test','train'};
files = dir(SOURCE_DATASET)
dirFlags = [files.isdir]
subFolders = files(dirFlags)

for f = 3 : length(subFolders)  
  fprintf('Sub folder #%d = %s\n', f, subFolders(f).name);
  for t=1: length(groups)
      for h=1:length(FPS_POINTS)
          mkdir(strcat('D:\Pottery_Result\pottery_geodesic_m',num2str(FPS_POINTS(h)),'\',subFolders(f).name,'\',groups{t}))
      end
      path = strcat(SOURCE_DATASET,subFolders(f).name,'\',groups{t},'\')
      myfiles = dir(strcat(path,'*.obj')) 
      for x=1:length(myfiles)
        clf;%clear figure
        name = strcat(path,myfiles(x).name);
        [pathstr,file_name,ext] = fileparts(name);        
        [vertex,faces] = read_obj(name);
        %get min or max vertex of Y_axis as initial point
        Y_axis = vertex(2,:);
        if strcmp(VERTEX_ORIGIN_CRITERION,'min')
            [val, ORIGIN_VERTEX] = min(Y_axis);
        else
            [val, ORIGIN_VERTEX] = max(Y_axis);
        end
        landmarks = [ORIGIN_VERTEX];
        [D,Z,Q] = perform_fast_marching_mesh(vertex, faces, landmarks);
        k = 1;

        for i=1:m         
            if i==FPS_POINTS(k)
                hl = camlight('headlight');
                options.face_vertex_color = perform_hist_eq(D,'linear');
                plot_mesh(vertex,faces, options);
                colormap jet(256);
                zoom out;
                zoom(1.0);
                %set(gca,'cameraviewanglemode','manual');%Neccesary for have the same zoom when the object is rotated
                hold on;
                %To show the FPS POINTS over the 3D object
                %h = plot3(vertex(1,landmarks), vertex(2,landmarks),vertex(3,landmarks), 'r.');set(h, 'MarkerSize', 25); 
                views = 12
                step = 360/views
                for v=1:views
                    I = frame2im(getframe(gcf)); %Convert plot to image (true color RGB matrix).
                    J = imresize(I, [224, 224], 'bicubic'); %Resize image to resolution
                    imwrite(J, strcat('D:\Pottery_Result\pottery_geodesic_m',num2str(FPS_POINTS(k)),'\',subFolders(f).name,'\',groups{t},'\',file_name,'_v',num2str(v-1),'.png')); %Save image to file
                    camorbit(-step,0,'coordsys',[0 1 0]);
                    camlight(hl,'headlight');  
                end
                k = k+1;
                delete(findall(gcf,'Type','light'));
            end
            %update distances
            [tmp,landmarks(end+1)] = max(D);
            options.constraint_map = D;
            [D1,Z,Q] = perform_fast_marching_mesh(vertex, faces, landmarks,options);
            D = min(D,D1);
        end
      end
  end
end