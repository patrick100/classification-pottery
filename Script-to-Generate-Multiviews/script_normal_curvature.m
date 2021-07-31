%Add libraries
cd 'C:\Users\jefri\Documents\toolbox_graph\toolbox_fast_marching';
path(path,'C:\Users\jefri\Documents\toolbox_graph\toolbox_signal');
path(path,'C:\Users\jefri\Documents\toolbox_graph\toolbox_graph');
path(path,'C:\Users\jefri\Documents\toolbox_graph\toolbox_fast_marching');
path(path, 'toolbox/');

%Type = 'normal';
Type = 'curvature';
%SOURCE_DATASET = 'D:\Pottery_Dataset\';
SOURCE_DATASET = 'D:\Peruvian_Dataset\';
if strcmp(Type,'normal')
    folder_output = 'D:\Pottery_Normal'
else
    folder_output = 'D:\Pottery_Curvature'
end

mkdir(folder_output);%output folder
groups = {'test','train'};
files = dir(SOURCE_DATASET)
dirFlags = [files.isdir]
subFolders = files(dirFlags)

for f = 3 : length(subFolders)  
  fprintf('Sub folder #%d = %s\n', f, subFolders(f).name);
  for t=1: length(groups)
      mkdir(strcat(folder_output,'\',subFolders(f).name,'\',groups{t}))
      path = strcat(SOURCE_DATASET,subFolders(f).name,'\',groups{t},'\')
      myfiles = dir(strcat(path,'*.obj')) 
      for x=1:length(myfiles)
        clf;%clear figure
        name = strcat(path,myfiles(x).name);
        [pathstr,file_name,ext] = fileparts(name);        
        [vertex,faces] = read_obj(name);
        if strcmp(Type,'normal')
            options.face_vertex_color = [];
            plot_mesh(vertex,faces, options); 
        else
            %compute the curvature
            options.curvature_smoothing = 10;
            options.verb = 0;
            [Umin,Umax,Cmin,Cmax,Cmean,Cgauss,Normal] = compute_curvature(vertex,faces,options);
            options.face_vertex_color = perform_saturation(Cmean,1.0);%mean curvature
            plot_mesh(vertex,faces, options); 
            colormap jet(256);
        end
        zoom out;
        zoom(1.0);
        hl = camlight('headlight');
        set(gca,'cameraviewanglemode','manual');%Neccesary for have the same zoom when the object is rotated
        hold on;
        views = 12
        step = 360/views
        for v=1:views
            I = frame2im(getframe(gcf)); %Convert plot to image (true color RGB matrix).
            J = imresize(I, [224, 224], 'bicubic'); %Resize image to resolution 320x240
            imwrite(J, strcat(folder_output,'\',subFolders(f).name,'\',groups{t},'\',file_name,'_v',num2str(v-1),'.png')); %Save image to file
            camorbit(-step,0,'coordsys',[0 1 0]);
            camlight(hl,'headlight');  
        end
        delete(findall(gcf,'Type','light'));
      end
  end
end